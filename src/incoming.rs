use futures::{task, Async, Future, Poll, Stream};
use futures::unsync::oneshot;
use mpi::point_to_point::{Message, Source, Status};
use void::Void;
use super::buffer::Unanchor;
use super::codec::{Codec, RecvInto};
use super::request_poll::RequestPoll;
use super::switch::Link;

/// Represents a stream of incoming messages.
///
/// ```ignore
/// Incoming<Source, Codec>: Stream<Future<(Status, Codec::Message)>>
/// ```
#[derive(Debug)]
#[must_use = "streams do nothing unless polled"]
pub struct Incoming<'a, C: Codec<'a>, S: Source> {
    link: Link<'a>,
    codec: C,
    source: S,
}

impl<'a, C: Codec<'a>, S: Source> Incoming<'a, C, S> {
    pub fn new(link: Link<'a>, codec: C, source: S) -> Self {
        Self {
            link: link.clone(),
            codec: codec,
            source: source,
        }
    }
}

impl<'a, C: Codec<'a>, S: Source> Stream for Incoming<'a, C, S> {
    type Item = WithStatus<C::FutureMessage>;
    type Error = Void;
    fn poll(&mut self) -> Poll<Option<Self::Item>, Self::Error> {
        self.link.modify_request_poll(|request_poll| match request_poll {
            None => Ok(Async::Ready(None)),
            Some(request_poll) => match self.source.immediate_matched_probe() {
                Some((msg, status)) => {
                    let recv_into = RecvIntoImpl {
                        request_poll: request_poll,
                        msg: msg,
                    };
                    let ((), fut_msg) = self.codec.decode(status, recv_into);
                    Ok(Async::Ready(Some(WithStatus(status, fut_msg))))
                }
                None => {
                    task::park().unpark();
                    Ok(Async::NotReady)
                }
            },
        })
    }
}

// FutureBuffer needs to be its own concrete type because associated type
// constructors don't exist yet :(
pub struct FutureBuffer<B>(oneshot::Receiver<B>);

impl<B> Future for FutureBuffer<B> {
    type Item = B;
    type Error = Void;
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.0.poll() {
            Err(oneshot::Canceled) => panic!("sender cancelled"),
            Ok(r) => Ok(r),
        }
    }
}

struct RecvIntoImpl<'b, 'a: 'b> {
    request_poll: &'b mut RequestPoll<'a>,
    msg: Message,
}

impl<'b, 'a> RecvInto<'a> for RecvIntoImpl<'b, 'a> {
    // we don't really use the Output type for anything but we keep it in the
    // trait anyway to enforce some sanity in the implementation of Codec
    type Output = ();
    fn recv_into<B: Unanchor + 'a>(self, buf: B)
                                   -> (Self::Output, FutureBuffer<B>) {
        let (sender, receiver) = oneshot::channel();
        self.request_poll.mrecv(self.msg, buf, move |anchor| {
            let _ = sender.send(B::unanchor(anchor));
        });
        ((), FutureBuffer(receiver))
    }
}

/// Used to wrap each received message with its `Status` information.
///
/// ```ignore
/// WithStatus<Future<T, E>>: Future<(Status, T), E>
/// ```
pub struct WithStatus<F: Future>(pub Status, pub F);

impl<F: Future> Future for WithStatus<F> {
    type Item = (Status, F::Item);
    type Error = F::Error;
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.1.poll() {
            Err(err) => Err(err),
            Ok(Async::NotReady) => Ok(Async::NotReady),
            Ok(Async::Ready(item)) => Ok(Async::Ready((self.0, item))),
        }
    }
}
