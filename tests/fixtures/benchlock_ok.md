# Fixture: every gated number agrees

Peak memory at n=1000 is <!--bench:mem.n1000_linear_bytes tol=10%-->146,216<!--/bench-->
bytes for the linear aligner and
<!--bench:mem.n1000_quadratic_bytes tol=10%-->24,167,696<!--/bench--> bytes for the
oracle, a ratio of <!--bench:ratio.n1000 tol=5%-->165.3<!--/bench-->x over
<!--bench:cells.n1000-->1,002,001<!--/bench--> cells, both scoring
<!--bench:score.n1000_linear-->1,387<!--/bench-->.
The differential sweep runs <!--bench:pairs.ci-->6,000<!--/bench--> pairs.
