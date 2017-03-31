extern crate futures;
extern crate mpi;
extern crate mpi_futures;
extern crate synchrotron;

use futures::{Future, Stream};
use mpi::topology::Communicator;
use mpi_futures::switch::Switch;
use mpi_futures::codec::U8Codec;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let comm = world.duplicate();
    let mut core = synchrotron::Core::default();
    let switch = Switch::default();
    let link = switch.link().with_codec(U8Codec);
    let handle = core.handle();
    let my_rank = comm.rank();
    let comm_size = comm.size();
    let target_rank = (my_rank + 1) % comm_size;
    handle.spawn(switch);
    handle.spawn({
        link.send(comm.process_at_rank(target_rank),
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
