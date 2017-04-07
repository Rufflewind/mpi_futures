//! The codec interface describes how messages are to be received and sent at
//! a low level.  In particular, it describes how a custom `Message` type is
//! to be mapped into an MPI datatype and vice versa.

use std::ops::DerefMut;
use conv::ValueInto;
use futures::Future;
use mpi::datatype::Equivalence;
use mpi::point_to_point::Status;
use void::Void;
use super::buffer::{OwnedBuffer, Unanchor};
use super::incoming::FutureBuffer;

// This trait is not unsafe to implement nor use.  Although the `Status` must
// be correctly associated with the message, this is meaningless in isolation
// as RecvInto objects are never created except as an implementation detail.
//
// Users don't have access to the concrete RecvIntoImpl and they have no way
// to observe the uninitialized state of the buffer (it can only occur after
// the receive has completed).
//
// The worst a user can do is to wrap an existing RecvInto, but that gives
// them nothing they can't already do!
pub trait RecvInto<'a>: Sized {

    type Output;

    // associate Status with the RecvInto object (we can't trust the user to
    // pass the correct Status when they call recv_into_vec)
    fn status(&self) -> &Status;

    fn recv_into<B>(self, buffer: B) -> (Self::Output, FutureBuffer<B>)
        where B: Unanchor + 'a;

    /// Convenience function if all you want is a simple `Vec`.
    fn recv_into_vec<T: Equivalence + 'a>(self) -> (Self::Output,
                                                    FutureBuffer<Vec<T>>) {
        let len = self.status()
            .count(T::equivalent_datatype()).value_into().unwrap();
        let mut buf = Vec::<T>::with_capacity(len);
        unsafe {
            buf.set_len(len);
        }
        self.recv_into(buf)
    }
}

pub trait SendFrom<'a> {
    type Output;

    fn send_from<B>(self, buffer: B, tag: u16) -> Self::Output
        where B: OwnedBuffer + 'a;
}

pub trait Decoder<'a> {
    type FutureMessage: Future<Error=Void>;

    fn decode<R: RecvInto<'a>>(&mut self, r: R)
                               -> (R::Output, Self::FutureMessage);
}

impl<'a, T: DerefMut<Target=U>, U: Decoder<'a>> Decoder<'a> for T {
    type FutureMessage = <T::Target as Decoder<'a>>::FutureMessage;

    fn decode<R: RecvInto<'a>>(&mut self, r: R)
                               -> (R::Output, Self::FutureMessage) {
        self.deref_mut().decode(r)
    }
}

pub trait Encoder<'a> {
    /// Type of each message produced by `Incoming` and consumed by `send`.
    ///
    /// (Not to be confused with `MPI_Message`, which is more of a handle to a
    /// pending message.)
    type Message;

    fn encode<S: SendFrom<'a>>(self, msg: Self::Message, s: S) -> S::Output;
}

/// Simple codec that simply treats every message as an array of octets and
/// always sets the tag to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct U8Codec;

impl<'a> Decoder<'a> for U8Codec {
    type FutureMessage = FutureBuffer<Vec<u8>>;

    fn decode<R: RecvInto<'a>>(&mut self, r: R)
                               -> (R::Output, Self::FutureMessage) {
        r.recv_into_vec::<u8>()
    }
}

impl<'a> Encoder<'a> for U8Codec {
    type Message = Vec<u8>;

    fn encode<S: SendFrom<'a>>(self, msg: Self::Message, s: S) -> S::Output {
        s.send_from(msg, 0)
    }
}
