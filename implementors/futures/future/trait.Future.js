(function() {var implementors = {};
implementors["futures"] = [];
implementors["mpi_futures"] = ["impl&lt;B&gt; <a class=\"trait\" href=\"futures/future/trait.Future.html\" title=\"trait futures::future::Future\">Future</a> for <a class=\"struct\" href=\"mpi_futures/incoming/struct.FutureBuffer.html\" title=\"struct mpi_futures::incoming::FutureBuffer\">FutureBuffer</a>&lt;B&gt;","impl&lt;F:&nbsp;<a class=\"trait\" href=\"futures/future/trait.Future.html\" title=\"trait futures::future::Future\">Future</a>&gt; <a class=\"trait\" href=\"futures/future/trait.Future.html\" title=\"trait futures::future::Future\">Future</a> for <a class=\"struct\" href=\"mpi_futures/incoming/struct.WithStatus.html\" title=\"struct mpi_futures::incoming::WithStatus\">WithStatus</a>&lt;F&gt;","impl&lt;'a, C:&nbsp;<a class=\"trait\" href=\"mpi_futures/codec/trait.Codec.html\" title=\"trait mpi_futures::codec::Codec\">Codec</a>&lt;'a&gt;, D:&nbsp;<a class=\"trait\" href=\"mpi/point_to_point/trait.Destination.html\" title=\"trait mpi::point_to_point::Destination\">Destination</a>&gt; <a class=\"trait\" href=\"futures/future/trait.Future.html\" title=\"trait futures::future::Future\">Future</a> for <a class=\"struct\" href=\"mpi_futures/send/struct.Send.html\" title=\"struct mpi_futures::send::Send\">Send</a>&lt;'a, C, D&gt;","impl&lt;'a, E&gt; <a class=\"trait\" href=\"futures/future/trait.Future.html\" title=\"trait futures::future::Future\">Future</a> for <a class=\"struct\" href=\"mpi_futures/switch/struct.Switch.html\" title=\"struct mpi_futures::switch::Switch\">Switch</a>&lt;'a, E&gt;",];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()
