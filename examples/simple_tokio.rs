// Variant of 'simple.rs' but uses tokio_core.  This one requires some
// workarounds because tokio_core doesn't allow arbitrary lifetimes in spawns.
extern crate futures;
extern crate mpi;
extern crate mpi_futures;
extern crate tokio_core;

use futures::{Future, Stream};
use mpi::topology::Communicator;
use mpi_futures::switch::Switch;
use mpi_futures::codec::U8Codec;

struct Process<C>(C, mpi::topology::Rank);

impl<C: Communicator> mpi::topology::AsCommunicator for Process<C> {
    type Out = C;
    fn as_communicator(&self) -> &Self::Out { &self.0 }
}

impl<C: Communicator> mpi::point_to_point::Destination for Process<C> {
    fn destination_rank(&self) -> mpi::topology::Rank { self.1 }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let mut core = tokio_core::reactor::Core::new().unwrap();
    let switch = Switch::default();
    let link = switch.link().with_codec(U8Codec);
    let handle = core.handle();
    let my_rank = comm.rank();
    let comm_size = comm.size();
    let target_rank = (my_rank + 1) % comm_size;
    handle.spawn(switch);
    handle.spawn({
        link.send(Process(comm, target_rank),
                  Vec::from(b"hello world" as &[u8]))
            .map(move |_| {
                println!("{}: sent to {}!", my_rank, target_rank)
            }).or_else(|_| {
                Ok(())
            })
    });
    core.run(
        link.incoming(comm.any_process())
            .buffered(1)
            .for_each(|(status, msg)| {
                println!("{}: received {:?} from {}",
                         my_rank,
                         String::from_utf8(msg).unwrap(),
                         status.source_rank());
                link.link.close();
                Ok(())
            })
    ).unwrap();
}
