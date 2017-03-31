//! The codec interface describes how messages are to be received and sent at
//! a low level.  In particular, it describes how a custom `Codec::Message`
//! type is to be mapped into an MPI datatype and vice versa.

use std::rc::Rc;
use std::sync::Arc;
use conv::ValueInto;
use futures::Future;
use mpi::datatype::Equivalence;
use mpi::point_to_point::Status;
use void::Void;
use super::buffer::{OwnedBuffer, Unanchor};
use super::incoming::FutureBuffer;

pub trait RecvInto<'a> {
    type Output;
    fn recv_into<B>(self, buffer: B) -> (Self::Output, FutureBuffer<B>)
        where B: Unanchor + 'a;
}

pub trait SendFrom<'a> {
    type Output;
    fn send_from<B>(self, buffer: B, tag: u16) -> Self::Output
        where B: OwnedBuffer + 'a;
}

pub trait Codec<'a> {
    /// Type of each message produced by `Incoming` and consumed by `send`.
    ///
    /// (Not to be confused with `MPI_Message`, which is more of a handle to a
    /// pending message.)
    type Message;
    type FutureMessage: Future<Item=Self::Message, Error=Void>;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage);
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S)
                               -> S::Output;
}

impl<'a, 'b, C: Codec<'a>> Codec<'a> for &'b C {
    type Message = C::Message;
    type FutureMessage = C::FutureMessage;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage) {
        (*self).decode(status, r)
    }
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S) -> S::Output {
        (*self).encode(msg, s)
    }
}

impl<'a, 'b, C: Codec<'a>> Codec<'a> for &'b mut C {
    type Message = C::Message;
    type FutureMessage = C::FutureMessage;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage) {
        (*self as &C).decode(status, r)
    }
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S) -> S::Output {
        (*self as &C).encode(msg, s)
    }
}

impl<'a, C: Codec<'a>> Codec<'a> for Rc<C> {
    type Message = C::Message;
    type FutureMessage = C::FutureMessage;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage) {
        (**self).decode(status, r)
    }
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S) -> S::Output {
        (**self).encode(msg, s)
    }
}

impl<'a, C: Codec<'a>> Codec<'a> for Arc<C> {
    type Message = C::Message;
    type FutureMessage = C::FutureMessage;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage) {
        (**self).decode(status, r)
    }
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S) -> S::Output {
        (**self).encode(msg, s)
    }
}

/// Simple codec that simply treats every message as an array of octets and
/// always sets the tag to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct U8Codec;

impl<'a> Codec<'a> for U8Codec {
    type Message = Vec<u8>;
    type FutureMessage = FutureBuffer<Vec<u8>>;
    fn decode<R: RecvInto<'a>>(&self, status: Status, r: R)
                               -> (R::Output, Self::FutureMessage) {
        let len = status.count(u8::equivalent_datatype()).value_into().unwrap();
        let mut buf = Vec::<u8>::with_capacity(len);
        unsafe {
            buf.set_len(len);
        }
        r.recv_into(buf)
    }
    fn encode<S: SendFrom<'a>>(&self, msg: Self::Message, s: S) -> S::Output {
        s.send_from(msg, 0)
    }
}
