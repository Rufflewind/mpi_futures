use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};
use futures::{Async, Future, Poll};
use futures::task;
use mpi::point_to_point::{Destination, Source};
use super::request_poll::RequestPoll;
use super::codec::Codec;
use super::incoming::Incoming;
use super::send::Send;

#[derive(Debug, Default)]
struct Inner<'a> {
    request_poll: RequestPoll<'a>,
    stop: bool,
}

/// Scheduler for MPI communications.
///
/// It can be constructed via `Switch::default()`.
///
/// A `Switch` is responsible for managing send and receive requests and
/// notifying tasks whenever they are ready.
///
/// As a `Future`, it intended to be spawned as a separate task on the
/// executor and will run continuously until `Link::close` is called.  If the
/// `Switch` is not running, any futures that are linked to this switch will
/// block forever.
#[derive(Debug)]
pub struct Switch<'a, E>(Rc<RefCell<Inner<'a>>>, PhantomData<E>);

impl<'a, E> Default for Switch<'a, E> {
    fn default() -> Self {
        Switch(Default::default(), Default::default())
    }
}

impl<'a, E> Switch<'a, E> {
    /// Acquire a `Link` to this `Switch`.  A `Link` acts as a clonable
    /// delegate for the switch and allows performing MPI requests.
    pub fn link(&self) -> Link<'a> {
        Link(Rc::downgrade(&self.0))
    }
}

impl<'a, E> Future for Switch<'a, E> {
    type Item = ();
    type Error = E;
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        let mut inner = self.0.borrow_mut();
        if inner.stop {
            Ok(Async::Ready(()))
        } else {
            inner.request_poll.test();
            task::park().unpark();
            Ok(Async::NotReady)
        }
    }
}

/// Used to perform MPI requests through a `Switch`.
///
/// Unlike `Switch`, which can't be cloned, `Link` can be cloned as many times
/// as you like and is always linked to same switch it originally came from.
#[derive(Debug, Clone)]
pub struct Link<'a>(Weak<RefCell<Inner<'a>>>);

impl<'a> Link<'a> {
    /// Gracefully shut down the associated `Switch`.  No more requests will
    /// be processed and any pending requests will be cancelled where possible
    /// and waited on.
    ///
    /// Calling `close` after the switch has already been closed has no
    /// effect.
    pub fn close(&self) {
        self.0.upgrade().map(|inner| {
            inner.borrow_mut().stop = true;
        });
    }

    /// Combine the `Link` with a `Codec` to create a `LinkedCodec`, which can
    /// be used to send and receive messages.
    pub fn with_codec<C>(self, codec: C) -> LinkedCodec<'a, C>
        where C: Codec<'a> + Clone
    {
        LinkedCodec {
            link: self,
            codec: codec,
        }
    }

    /// Modify the internal `RequestPoll`, if the `Switch` is still alive.
    /// This is mostly for internal use.  Nesting calls to this function will
    /// cause panics due to repeated borrows.
    pub fn modify_request_poll<F, R>(&self, f: F) -> R
        where F: FnOnce(Option<&mut RequestPoll<'a>>) -> R
    {
        match self.0.upgrade() {
            None => f(None),
            Some(inner) => f(Some(&mut inner.borrow_mut().request_poll)),
        }
    }
}

/// A combined `Link` and `Codec`.
#[derive(Debug, Clone)]
pub struct LinkedCodec<'a, C: Codec<'a> + Clone> {
    /// The `Link` to the `Switch`.
    pub link: Link<'a>,
    /// The message `Codec`.
    pub codec: C,
}

impl<'a, C: Codec<'a> + Clone> LinkedCodec<'a, C> {
    /// Obtain a `Stream` of future incoming messages from the given `source`.
    /// Each message is decoded using the given `codec`.
    ///
    /// ```ignore
    /// fn incoming(&self, Source) -> Stream<Future<Codec::Message>>;
    /// ```
    ///
    /// The stream will keep running until the `Switch` is `close`d, but you
    /// can stop the `Stream` at any time if you aren't expecting to receive
    /// messages.  You can even create a new `incoming` stream every time you
    /// want to receive a message.
    ///
    /// Just try to avoid running multiple overlapping `incoming` streams
    /// simultaneously, as that could cause messages to be split between the
    /// streams in a non-deterministic manner.
    pub fn incoming<S: Source>(&self, source: S) -> Incoming<'a, C, S> {

        Incoming::new(self.link.clone(), self.codec.clone(), source)
    }

    /// Send a message asynchronously, returning a `Future` that completes
    /// when the send does.
    ///
    /// ```ignore
    /// fn send(&self, Destination, Codec::Message) -> Future<()>;
    /// ```
    pub fn send<D: Destination>(&self, dest: D, msg: C::Message)
                                -> Send<'a, C, D> {
        Send::new(self.link.clone(), self.codec.clone(), dest, msg)
    }

    /// Same as [`Link::close`](struct.Link.html#method.close).
    pub fn close(&self) {
        self.link.close()
    }
}
