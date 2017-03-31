use std::{mem, slice};
use std::rc::Rc;
use std::sync::Arc;
use mpi::datatype::{Buffer, BufferMut, Equivalence};

/// An owned buffer that can be read from.
///
/// # Unsafe invariant
///
/// The `Buffer` returned by `as_buffer` must survive so long as `Self` is not
/// operated on in any way beyond moves.
pub unsafe trait OwnedBuffer {
    type Buffer: Buffer + ?Sized;
    fn as_buffer(&self) -> &Self::Buffer;
}

unsafe impl<'a, T: Equivalence> OwnedBuffer for &'a T {
    type Buffer = T;
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<'a, T: Equivalence> OwnedBuffer for &'a [T] {
    type Buffer = [T];
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<T: Equivalence> OwnedBuffer for Box<T> {
    type Buffer = T;
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<T: Equivalence> OwnedBuffer for Box<[T]> {
    type Buffer = [T];
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<T: Equivalence> OwnedBuffer for Vec<T> {
    type Buffer = [T];
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<T: Equivalence> OwnedBuffer for Rc<Vec<T>> {
    type Buffer = [T];
    fn as_buffer(&self) -> &Self::Buffer { self }
}

unsafe impl<T: Equivalence> OwnedBuffer for Arc<Vec<T>> {
    type Buffer = [T];
    fn as_buffer(&self) -> &Self::Buffer { self }
}

/// An owned buffer that can be modified.
pub trait OwnedBufferMut {
    type BufferMut: BufferMut + ?Sized;

    /// The anchor is a (possibly opaque) object that is responsible for
    /// performing clean up after the buffer is no longer needed.
    ///
    /// To avoid aliasing issues, the anchor is usually represented by a
    /// separate data type that contains raw pointers (rather than an owned
    /// data type like `Vec`, which internally contains a `Unique`).
    type Anchor;

    /// Split the owned buffer into an anchor and a mutable buffer.
    ///
    /// This function is unsafe because lifetime of the buffer is misleading:
    /// it will only last as long as the anchor survives.
    unsafe fn into_buffer_mut<'a>(self) -> (Self::Anchor,
                                            &'a mut Self::BufferMut);
}

// this is safe because by the time 'a ends we know the RequestPoll will have
// also been dropped; during that time, there is no other mutable alias of the
// buffer.
impl<'a, T: Equivalence> OwnedBufferMut for &'a mut T {
    type BufferMut = T;
    type Anchor = ();

    unsafe fn into_buffer_mut<'b>(self) -> (Self::Anchor,
                                            &'b mut Self::BufferMut) {
        ((), mem::transmute(self))
    }
}

impl<'a, T: Equivalence> OwnedBufferMut for &'a mut [T] {
    type BufferMut = [T];
    type Anchor = ();

    unsafe fn into_buffer_mut<'b>(self) -> (Self::Anchor,
                                            &'b mut Self::BufferMut) {
        ((), mem::transmute(self))
    }
}

/// Upgrade an anchor back into its original owned buffer.  Not all buffers
/// support this operation.
pub trait Unanchor: OwnedBufferMut {
    /// Convert the anchor back into its original form.
    ///
    /// This is safe because we assume that if safe code has access to
    /// an anchor, then it is no longer borrowed.
    fn unanchor(anchor: Self::Anchor) -> Self;
}

pub struct AnchoredBox<T: Equivalence>(*mut T);

impl<T: Equivalence> Drop for AnchoredBox<T> {
    fn drop(&mut self) {
        unsafe {
            Box::<T>::unanchor(mem::replace(self, mem::uninitialized()));
        }
    }
}

impl<T: Equivalence> OwnedBufferMut for Box<T> {
    type BufferMut = T;
    type Anchor = AnchoredBox<T>;

    unsafe fn into_buffer_mut<'a>(self) -> (Self::Anchor,
                                            &'a mut Self::BufferMut) {
        let ptr = Box::into_raw(self);
        let anchor = AnchoredBox(ptr);
        (anchor, &mut *ptr)
    }
}

impl<T: Equivalence> Unanchor for Box<T> {
    fn unanchor(anchor: Self::Anchor) -> Self {
        let ptr = anchor.0;
        // make sure you forget this or it will stack overflow!
        mem::forget(anchor);
        unsafe {
            Box::from_raw(ptr)
        }
    }
}

pub struct AnchoredBoxedSlice<T: Equivalence> {
    ptr: *mut T,
    len: usize,
}

impl<T: Equivalence> Drop for AnchoredBoxedSlice<T> {
    fn drop(&mut self) {
        unsafe {
            Box::<[T]>::unanchor(mem::replace(self, mem::uninitialized()));
        }
    }
}

impl<T: Equivalence> OwnedBufferMut for Box<[T]> {
    type BufferMut = [T];
    type Anchor = AnchoredBoxedSlice<T>;

    unsafe fn into_buffer_mut<'a>(mut self) -> (Self::Anchor,
                                                &'a mut Self::BufferMut) {
        let anchor = AnchoredBoxedSlice {
            ptr: self.as_mut_ptr(),
            len: self.len(),
        };
        mem::forget(self);
        let slice = slice::from_raw_parts_mut(anchor.ptr, anchor.len);
        (anchor, slice)
    }
}

impl<T: Equivalence> Unanchor for Box<[T]> {
    fn unanchor(anchor: Self::Anchor) -> Self {
        let orig = unsafe {
            Vec::from_raw_parts(anchor.ptr, anchor.len, anchor.len)
        };
        // make sure you forget this or it will stack overflow!
        mem::forget(anchor);
        orig.into_boxed_slice()
    }
}

// have to disassemble the vector to avoid aliasing violations;
// this signifies that the Vec is "locked" by the mut-borrow
pub struct AnchoredVec<T: Equivalence> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T: Equivalence> Drop for AnchoredVec<T> {
    fn drop(&mut self) {
        unsafe {
            Vec::unanchor(mem::replace(self, mem::uninitialized()));
        }
    }
}

impl<T: Equivalence> OwnedBufferMut for Vec<T> {
    type BufferMut = [T];
    type Anchor = AnchoredVec<T>;

    unsafe fn into_buffer_mut<'a>(mut self) -> (Self::Anchor,
                                                &'a mut Self::BufferMut) {
        let anchor = AnchoredVec {
            ptr: self.as_mut_ptr(),
            len: self.len(),
            capacity: self.capacity(),
        };
        mem::forget(self);
        let slice = slice::from_raw_parts_mut(anchor.ptr, anchor.len);
        (anchor, slice)
    }
}

impl<T: Equivalence> Unanchor for Vec<T> {
    fn unanchor(anchor: Self::Anchor) -> Self {
        let orig = unsafe {
            Vec::from_raw_parts(anchor.ptr, anchor.len, anchor.capacity)
        };
        // make sure you forget this or it will stack overflow!
        mem::forget(anchor);
        orig
    }
}
