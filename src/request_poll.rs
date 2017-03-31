//! An interface for polling multiple MPI requests simultaneously as well as
//! managing ownership of the associated buffers.

use std::{self, fmt, mem, ptr};
use conv::ValueInto;
use libc;
use mpi;
use mpi::raw::AsRaw;
use mpi::point_to_point::{Destination, Message};
use super::buffer::{OwnedBuffer, OwnedBufferMut};

fn abort(errorcode: libc::c_int) -> ! {
    unsafe {
        mpi::ffi::MPI_Abort(mpi::ffi::RSMPI_COMM_WORLD, errorcode);
        libc::abort();
    }
}

trait OrAbort {
    fn or_abort(self);
}

impl OrAbort for libc::c_int {
    fn or_abort(self) {
        if self != 0 {
            abort(self);
        }
    }
}

unsafe fn unbind_buffer<'a, B: OwnedBuffer>(b: &B) -> &'a B::Buffer {
    mem::transmute(b.as_buffer())
}

trait Callback {
    fn callback(self: Box<Self>) {}
}

struct CallbackImpl<F>(F);

impl<F: FnOnce()> Callback for CallbackImpl<F> {
    fn callback(self: Box<Self>) {
        self.0()
    }
}

/// Manages a collection of requests and keeps their associated buffers alive.
///
/// When `RequestPoll` is dropped, all pending requests will be canceled when
/// possible and waited on.
pub struct RequestPoll<'a> {
    // These three vectors are all synchronized in length and position of
    // items.  Every callback must outlive its corresponding MPI_Request,
    // because within the callback's context there is an anchor that is
    // responsible for keeping the buffer alive.
    requests: Vec<mpi::ffi::MPI_Request>,
    cancelables: Vec<bool>,
    callbacks: Vec<Box<Callback + 'a>>,

    // Temporary caches for indices from the previous test.  (Don't bother
    // with Statuses because the information is not useful for sends, and for
    // receives we're already probing anyway.)
    indices: Vec<libc::c_int>,
}

impl<'a> fmt::Debug for RequestPoll<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let callbacks: Vec<_> =
            self.callbacks.iter().map(|f| f as *const _).collect();
        f.debug_struct("RequestPoll")
            .field("requests", &self.requests)
            .field("cancelables", &self.cancelables)
            .field("callbacks", &callbacks)
            .field("indices", &self.indices)
            .finish()
    }
}

impl<'a> Default for RequestPoll<'a> {
    fn default() -> Self {
        Self {
            requests: Default::default(),
            cancelables: Default::default(),
            callbacks: Default::default(),
            indices: Default::default(),
        }
    }
}

/// Note: null requests are allowed.
unsafe fn wait_all(requests: &mut [mpi::ffi::MPI_Request]) {
    // fortunately, MPI_Waitall ignores null requests \o/
    let mut remaining = requests.len();
    while let Some(offset) = remaining.checked_sub(0xffff) {
        mpi::ffi::MPI_Waitall(
            0xffff,
            requests.as_mut_ptr().offset(offset as _),
            mpi::ffi::RSMPI_STATUSES_IGNORE).or_abort();
        remaining = offset;
    }
    mpi::ffi::MPI_Waitall(
        remaining as _,
        requests.as_mut_ptr(),
        mpi::ffi::RSMPI_STATUSES_IGNORE).or_abort();
}

/// Note: null requests are allowed.
unsafe fn free_all(requests: &mut [mpi::ffi::MPI_Request]) {
    for request in requests {
        if *request != mpi::ffi::RSMPI_REQUEST_NULL {
            mpi::ffi::MPI_Request_free(request).or_abort();
        }
    }
}

impl<'a> Drop for RequestPoll<'a> {
    fn drop(&mut self) {
        unsafe {
            // Must wait on all the requests otherwise MPI might access freed
            // memory.  Try to cancel them first though, because otherwise the
            // wait might block forever.
            //
            // NOTE: Cancel on send is going to be phased out by the specs.
            for (request, &cancelable) in self.requests.iter_mut()
                                               .zip(&self.cancelables) {
                if *request != mpi::ffi::RSMPI_REQUEST_NULL && cancelable {
                    mpi::ffi::MPI_Cancel(request).or_abort();
                }
            }
            // deactivate all requests and free all non-persistent ones
            wait_all(&mut self.requests);
            // free remaining persistent requests
            free_all(&mut self.requests);
        }
        // (the anchors in self.callbacks will get freed automatically)
    }
}

impl<'a> RequestPoll<'a> {
    fn flush(&mut self) {
        // first pull out the request data without removing anything: we must
        // not swap_remove the other Vecs because the ordering of self.indices
        // is unknown
        for &i in &self.indices {
            let i = i as usize;
            // call the callbacks in the original order of the indices
            unsafe {
                ptr::read(&self.callbacks[i]).callback();
            }
        }
        // sort the indices so we can clean up the other Vecs
        self.indices.sort();
        for i in self.indices.drain(..).rev() {
            let i = i as _;
            self.cancelables.swap_remove(i);
            // don't drop it because we already called it!
            mem::forget(self.callbacks.swap_remove(i));
            // remove and free the request if it's persistent
            let mut request = self.requests.swap_remove(i);
            unsafe {
                if request != mpi::ffi::RSMPI_REQUEST_NULL {
                    mpi::ffi::MPI_Request_free(&mut request).or_abort();
                }
            }
        }
    }

    fn poll_with<F>(&mut self, f: F)
        where F: FnOnce(libc::c_int, *mut mpi::ffi::MPI_Request,
                        *mut libc::c_int, *mut libc::c_int,
                        *mut mpi::ffi::MPI_Status) -> libc::c_int
    {
        if self.requests.is_empty() {
            // MPI does stupid things when the request list is empty
            return;
        }
        let incount = self.requests.len();
        self.indices.reserve(incount);
        let incount = incount.value_into().unwrap(); // may panic
        unsafe {
            let mut outcount: libc::c_int = mem::uninitialized();
            f(incount,
              self.requests.as_mut_ptr(),
              &mut outcount,
              self.indices.as_mut_ptr(),
              mpi::ffi::RSMPI_STATUSES_IGNORE).or_abort();
            let outcount = outcount as _;
            debug_assert!(outcount <= self.indices.capacity());
            self.indices.set_len(outcount);
        }
    }

    /// Non-blocking test to see if some of the requests have completed.  For
    /// any request that is complete, the corresponding callback is called.
    pub fn test(&mut self) {
        self.poll_with(|n, r, m, i, s| unsafe {
            mpi::ffi::MPI_Testsome(n, r, m, i, s)
        });
        self.flush();
    }

    /// Block until at least one request has completed.  Otherwise functions
    /// similar to `test`.
    pub fn wait(&mut self) {
        self.poll_with(|n, r, m, i, s| unsafe {
            mpi::ffi::MPI_Waitsome(n, r, m, i, s)
        });
        self.flush();
    }

    /// Perform a matched receive on a message.
    pub fn mrecv<B, F>(&mut self, msg: Message, buf: B, callback: F)
        where B: OwnedBufferMut,
              B::Anchor: 'a,
              F: FnOnce(B::Anchor) + 'a,
    {
        self.reserve_one();             // may panic
        unsafe {
            let (anchor, buf) = buf.into_buffer_mut();
            let request = msg.immediate_matched_receive_into(buf);
            let callback = move || callback(anchor);
            self.insert(request.as_raw(), callback, true);
            mem::forget(request);
        }
    }

    /// Send a message.
    pub fn send<D, B, F>(&mut self, dest: D, buf: B,
                         tag: u16, callback: F)
        where D: Destination,
              B: OwnedBuffer + 'a,
              F: FnOnce(B) + 'a,
    {
        self.reserve_one();             // may panic
        // u16 is used here to prevent going over MPI_TAG_UB
        // (the minimum MPI_TAG_UB is 2^15 - 1, but most MPI implementations
        // support much more than that)
        let tag = tag.value_into().unwrap();
        unsafe {
            let buf_ref = unbind_buffer(&buf);
            let request = dest.immediate_send_with_tag(buf_ref, tag);
            let callback = move || callback(buf);
            self.insert(request.as_raw(), callback, false);
            std::mem::forget(request);
        }
    }

    /// Insert a request to be monitored.
    ///
    /// `cancelable` indicates whether `MPI_Cancel` will work on the request
    /// (`true` for receiving requests, `false` for all other requests).
    ///
    /// # Unsafety
    ///
    /// The request must be valid (in particular, not `MPI_REQUEST_NULL`).
    /// The buffers associated with the request must survive so long as the
    /// callback remains alive.
    pub unsafe fn insert<F>(&mut self, request: mpi::ffi::MPI_Request,
                            callback: F, cancelable: bool)
        where F: FnOnce() + 'a
    {
        self.requests.push(request);
        self.cancelables.push(cancelable);
        self.callbacks.push(Box::new(CallbackImpl(callback)));
    }

    /// Allocate room for a single request if necessary.
    ///
    /// # Panics
    ///
    /// Panics if the internal capacity overflows `usize`.
    pub fn reserve_one(&mut self) {
        self.requests.reserve(1);
        self.cancelables.reserve(1);
        self.callbacks.reserve(1);
    }
}
