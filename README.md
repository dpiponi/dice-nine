TODO
----
* eliminate context, move global into interpreter
* keyword args at top
* Maybe starred expressions like [\*(2 @ d(6)), \*(3 @ d(8))]
* cache preanalysis results
* maybe eliminate while
* does loop preanalysis really work?
* deal with fact that vars unreferenced in if/else branch may need del.
* make sure analysis understands del
* allow singletons in ranges, eg. x[a:b, c, d:e]. (make sure step of 1 is default.)
* When marginalizing we might need to leave the sum over all probs in the factor.
* make sure (A and B and C) works etc.
* make concat friendlier
* Use cuboid packing and then fall back to a sequence of hashes.
* Support float*, complex*.
* append()
* Return tuples from functions.
* argument for __chop__
* AugAssign in generators
* Find a use for ... :)

DONE
----
* x[i] += ..., ...
* NOT FULLZY IMPLEMENTED: support proper variable slices in reads of a[...]
* support a[...] = writes on tf and pytorch
* add len()
* change focus of debug to state changes rather than state
* restructured tests
* uses environment variable PL\_PLATFORM to choose platform
* error report needs to point to correct line
* NOT IMPLEMENTING: Maybe `and` of generators. So 2 @ d(6) and 3 @ d(8)?
* Maybe consider {} for generators so {d(6), d(8), ...} is a generator
  - this would make {\*(2 @ d(6)), \*(3 @ d(8))} an alternative to `and` above.
* how can I use set notation?
  - Answer: streams!
* avoid polluting namespace when using problib
* Tidy up visit\_Call
* switch to using logging rather than config.debug
* constant([]) bug
* reduce\_sum needs more flexibility in its axis.
* avoid polluting namespace when using problib
* more functions into lib
* support x[i] = ...
* vararg user functions
* preanalysis needs to work with called functions
* Support passing in constants at least at top level.
* preanalysis needs to work with arguments
* support "ranges" and maybe notation like range[1:3,4,8:10]???
* flip
* Remove need for move()
* program analysis to eliminate need for del and move
* Need bitcast.
* consider separating variables from underlying registers with
  mapping from variables to registers.
* generators!
* += etc.
* top k
* run(..., squeeze=True)
* Delay conversion to tensor so ints, say, remain Python ints.
  - If a variable is a certainty then maybe it can be kept independent.
