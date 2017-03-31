use std::{fmt, mem};
use futures::{Async, Future, Poll};
use futures::unsync::oneshot;
use mpi::point_to_point::Destination;
use void::Void;
use super::buffer::OwnedBuffer;
use super::codec::{Codec, SendFrom};
use super::request_poll::RequestPoll;
use super::switch::Link;

enum State<'a, C: Codec<'a>, D> {
    Pending {
        link: Link<'a>,
        codec: C,
        dest: D,
        msg: C::Message,
    },
    Started {
        receiver: oneshot::Receiver<()>,
    },
    Invalid,
}

impl<'a, C, D> fmt::Debug for State<'a, C, D>
    where C: Codec<'a> + fmt::Debug,
          C::Message: fmt::Debug,
          D: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &State::Pending { ref link, ref codec, ref dest, ref msg } =>
                f.debug_struct("State::Pending")
                .field("link", link)
                .field("codec", codec)
                .field("dest", dest)
                .field("msg", msg)
                .finish(),
            &State::Started { ref receiver } =>
                f.debug_struct("State::Started")
                .field("receiver", receiver)
                .finish(),
            &State::Invalid =>
                f.write_str("State::Invalid"),
        }
    }
}

pub struct Send<'a, C: Codec<'a>, D>(State<'a, C, D>);

impl<'a, C, D> fmt::Debug for Send<'a, C, D>
    where C: Codec<'a> + fmt::Debug,
          C::Message: fmt::Debug,
          D: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Send")
            .field(&self.0)
            .finish()
    }
}

impl<'a, C: Codec<'a>, D: Destination> Send<'a, C, D> {
    pub fn new(link: Link<'a>, codec: C, dest: D, msg: C::Message) -> Self {
        Send(State::Pending {
            link: link,
            codec: codec,
            dest: dest,
            msg: msg,
        })
    }
}

struct SendFromImpl<'b, 'a: 'b, D> {
    request_poll: &'b mut RequestPoll<'a>,
    dest: D,
    sender: oneshot::Sender<()>,
}

impl<'b, 'a, D: Destination> SendFrom<'a> for SendFromImpl<'b, 'a, D> {
    // we don't really use the Output type for anything but we keep it in the
    // trait anyway to enforce some sanity in the implementation of Codec
    type Output = ();
    fn send_from<B: OwnedBuffer + 'a>(self, buf: B, tag: u16)
                                      -> Self::Output {
        let sender = self.sender;
        self.request_poll.send(self.dest, buf, tag, move |_| {
            let _ = sender.send(());
        });
    }
}

impl<'a, C: Codec<'a>, D: Destination> Future for Send<'a, C, D> {
    type Item = ();
    type Error = Void;
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        fn poll_receiver<F: Future<Item=()>>(receiver: &mut F)
                                             -> Poll<(), Void> {
            Ok(receiver.poll().unwrap_or(Async::Ready(())))
        }
        match mem::replace(&mut self.0, State::Invalid) {
            State::Pending { link, codec, dest, msg } =>
                link.modify_request_poll(|request_poll| match request_poll {
                    None => Ok(Async::Ready(())),
                    Some(request_poll) => {
                        let (sender, mut receiver) = oneshot::channel();
                        let send_from = SendFromImpl {
                            request_poll: request_poll,
                            dest: dest,
                            sender: sender,
                        };
                        codec.encode(msg, send_from);
                        let poll = poll_receiver(&mut receiver);
                        self.0 = State::Started { receiver: receiver };
                        poll
                    }
                }),
            State::Started { mut receiver } => poll_receiver(&mut receiver),
            // panic loudly so the loop doesn't just silently stall!
            State::Invalid => panic!("invalid state"),
        }
    }
}
