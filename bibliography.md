# Sampling without replacement

I have mostly researched algorithms that operate in `O(1)` space, since these
are the most practical for use on a GPU. There's a number of alternatives which
use auxiliary datastructures (usually a hash set).

- ["Faster Methods for Random Sampling" Vitter][vitter1984]

    This paper describes a few sequential, `O(1)` space, sampling algorithms,
    building up on one another.

    The first one to be presented, *Algorithm S*, iterates over every input
    element, selecting it with a propability `k/n`, where `k` and `n` are
    the number of outputs and inputs *remaining*. This algorithm runs in
    `O(n)`, making it very inefficient if `k` is much smaller than `n`.
    Nevertheless this algorithm is frequently used as the base case for
    other implementations that can reduce the size of `n`.

    The final entry and primary contribution of the paper, *Algorithm D*, runs
    in `O(k)` average time (where `k` is the size of the output), by drawing
    the size of the gaps between items to be sampled. It uses rejection
    sampling to draw the discrete values from a continuous approximation.

- ["Sequential Random Sampling" Ahrens et al.][ahrens1985]

    This paper proposes an algorithm based on a bernoulli sampling, where every
    element has a fixed probability of being selected. Generating a bernoulli
    sample is easy by using the geometric distribution to draw the gap sizes
    between selected elements.

    On average, a bernoulli sample of the input with probability `k / n` will
    produce `k` elements. In practice however, it will more often than not
    produce a couple extra items, or too few. Given `q` the number of selected
    elements by the bernoulli sample, if `q > k` then `q - k` elements can be
    removed by performing a recursive sample without replacement. If `q < k`,
    the set of selected elements is abandoned and the process starts again. By
    using a probability slightly greater than `k / n`, we can err on the side
    of having too many elements more often than we have too few.

    The basic algorithm (*Algorithm SG*) needs temporary storage to keep the
    `q` elements while it decides which ones to remove. A variant of it
    (*Algorithm SG\**) uses a deterministic PRNG to perform the same bernoulli
    sample twice: on the first run, the individual items are discarded and only
    the item count `q` is calculated.  The second run is combined with the
    nested sample-without-replacement of the `q - k` elements to remove,
    returning items that appear only in the bernoulli sample (both samples are
    produced in-order, making it easy to join them).

- ["An Efficient Algorithm for Sequential Random Sampling" Vitter][vitter1987]

    Vitter has another look at *Algorithm D* from his earlier paper. The paper
    "reaffirm[s] the efficiency and practicality of Method D and present[s]
    some implementation improvements". The paper includes benchmarks comparing
    its performance to Ahrens' SG and SG\* algorithms, and claims that
    *Algorithm D* is the fastest.  It also includes a Pascal implementation of
    the algorithm.

- ["Efficient Parallel Random Sampling—Vectorized, Cache-Efficient, and Online" Sanders et al.][sanders2018]

    A method to partition the sampling process across parallel "processors". It
    proposes a divide and conquer scheme, recursively splitting the input in
    half and processing each half independently and in parallel. The number of
    output items to get from the left or from the right half can be drawn from
    a hypergeometric distribution. Eventually the scheme needs a non-parallel
    algorithm as its base case, unless one recurses until `k=1`, which can be
    solved trivially.

    Rather than require coordination across the processors to communicate the
    variates of the hypergeometric distribution, the authors suggest using
    deterministic PRNG, with all processors sharing the same RNG state until
    their execution diverges. Once execution has diverged, the RNGs must be
    independent. This can be achieve by combining the raw RNG output together
    with the sub-range of the input being processed, using a "high-quality
    hash" function.

    The paper includes a nice survey of existing methods, including a
    suggestion for implementating Ahrens' *Algorithm SG* on a GPU: the initial
    bernoulli sample is done in parallel on the GPU, and the result is
    "repaired" on the CPU using *Algorithm S*.

- ["Simple, Optimal Algorithms for Random Sampling Without Replacement" Ting][ting2021]

    Ting revisits Vitter's *Algorithm D*, and concludes that the custom
    distribution that the original algorithm used is actually the beta-binomial
    distribution, which in turn can be implemented using only the uniform and
    binomial distributions. This makes implementating the algorithm much more
    straightforward, given that these two distributions are widely available
    already.

- ["Sequential Random Sampling Revisited: Hidden Shuffle Method" Shekelyan et al.][shekelyan2021]

    A completely different take on the sequential sampling without replacement
    problem. Rather than thinking about the size of gaps between sampled items,
    the algorithm simulates a Knuth Shuffle of the input, after which the first
    `k` elements of the input are returned. No shuffle actually takes place:
    the algorithm predicts what the effect of the shuffle would have been and
    picks the elements in a way that mirrors those effects.

    The base case of the algorithm needs another sample without replacement,
    but the input size is reduced such that `k` is on the same order as `n`,
    making some of the more naive algorithms applicable.

    The algorithm is very impressive and uses a couple of clever statistical
    tricks. An easy to read Python implementation is included in the text of
    the paper. The authors claim better performance than *Algorithm D* (as well
    as other algorithms commonly used in mainstream languages' standard
    libraries).

    The Ting and Shekelyan papers were published around the same time, and
    neither one cites or benchmarks against the other one. Interestingly, both
    papers attribute the lack of adoption of Vitter's *Algorithm D* to the
    required complexity of its implementation.

- ["Algorithms for generating small random samples" Cicirello][cicirello2025]

    Short functions for generating 2 and 3 samples from a large input in
    constant time. The method generalizes to larger values of `k`, but each
    value of `k` requires its own function and the number of operations grows
    with `k^2`.

    If we use sampling without replacement to pick the infectees of a given
    infector, the number of individuals to pick will likely be, on average,
    very small. Using this algorithm as a fast path may be beneficial.

[vitter1984]: https://dl.acm.org/doi/pdf/10.1145/358105.893
[ahrens1985]: https://dl.acm.org/doi/pdf/10.1145/214392.214402
[vitter1987]: https://www.ittc.ku.edu/~jsv/Papers/Vit87.RandomSampling.pdf
[sanders2018]: https://dl.acm.org/doi/10.1145/3157734
[ting2021]: https://arxiv.org/pdf/2104.05091
[shekelyan2021]: https://www.dimacs.rutgers.edu/~graham/pubs/papers/hiddenshuffle.pdf
[cicirello2025]: https://onlinelibrary.wiley.com/doi/pdf/10.1002/spe.3379

# Weighted random sampling

- ["Weighted Random Sampling on GPUs" Lehmann et al][lehmann2022]
- ["Parallel Weighted Random Sampling" Hübschle-Schneider et al][hubschle2022]

[lehmann2022]: https://arxiv.org/pdf/2106.12270
[hubschle2022]: https://dl.acm.org/doi/pdf/10.1145/3549934

# Compressed bitsets

- ["Better bitmap performance with Roaring bitmaps" Chambi et al][chambi2014]

    Most exisiting bitset compression schemes use a run-length encoding, where
    a contiguous sequence of homogenous bits is replaced with a count of the
    bits. This gives good compression, but makes random access difficult.

    Roaring bitsets use a two-level indexing data structure. The space is
    partitioned into chunks, each covering 64k. The top level index is an
    ordered array, pointing to the secondary containers. Depending on their
    cardinality, the containers can have two different representations.
    Containers with fewer than 4096 elements (TODO: or is it 4095? 4097?) are
    stored as an ordered array of integers. Containers with more elements than
    that are stored using bitmaps.

- ["Compressed bitmap indexes: beyond unions and intersections" Kaser et al.][kaser2014]
- ["Consistently faster and smaller compressed bitmaps with Roaring" Lemire et al.][lemire2016]
- ["Roaring Bitmaps: Implementation of an Optimized Software Library" Lemire et al.][lemire2017]
- ["GPU-WAH: Applying GPUs to Compressing Bitmap Indexes with Word Aligned Hybrid" Andrzejewski et al.][andrzejewski2010]
- ["GPU-PLWAH: GPU-based implementation of the PLWAH algorithm for compressing bitmaps" by Andrzejewski et al][andrzejewski2011]
- ["Parallel acceleration of CPU and GPU range queries over large data sets" Nelson et al.][nelson2020]

[chambi2014]: https://arxiv.org/pdf/1402.6407
[kaser2014]: https://arxiv.org/pdf/1402.4466
[lemire2016]: https://arxiv.org/pdf/1603.06549
[lemire2017]: https://arxiv.org/pdf/1709.07821
[andrzejewski2010]: https://www.researchgate.net/profile/Robert-Wrembel/publication/221464250_GPU-WAH_Applying_GPUs_to_Compressing_Bitmap_Indexes_with_Word_Aligned_Hybrid/links/56cf5bf108ae059e375971b8/GPU-WAH-Applying-GPUs-to-Compressing-Bitmap-Indexes-with-Word-Aligned-Hybrid.pdf
[andrzejewski2011]: https://bibliotekanauki.pl/articles/206057.pdf
[nelson2020]: https://journalofcloudcomputing.springeropen.com/counter/pdf/10.1186/s13677-020-00191-w.pdf

# Fixed-radius near neighbors

- ["Improved GPU Near Neighbours Performance for Multi-Agent Simulations" Chisholm et al][chisholm2019]

[chisholm2019]: https://eprints.whiterose.ac.uk/id/eprint/153625/7/1-s2.0-S0743731519301340-main.pdf

# Miscellaneous

- ["Larger GPU-accelerated Brain Simulations with Procedural Connectivity" Knight et al.][knight2020]

[knight2020]: https://www.biorxiv.org/content/10.1101/2020.04.27.063693v2.full.pdf
