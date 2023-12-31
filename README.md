# Constraining Dark Energy with Baryon Acoustic Oscillations (v2)

## Context
-----------------------------------------------------------------------
This is a reprise of a project I completed in my third year of university.
I intended to only briefly revisit it, but the original directory was a
mess, I had moved from using `nbodykit` (which no longer runs on my 
laptop) to `Triumvirate` for power-/bispectrum estimation, and after another
year of cosmology I was much better equipped to appreciate and execute
it -- so I decided to redo it.

## Current Issues
-----------------------------------------------------------------------
The models for the power spectrum are surprisingly tricky to implement
without `nbodykit`. I have created a tiny, basic package called `cosmotools` that performs 
transfer function and linear power spectrum calculations very similarly to `nbodykit` but implements the cosmology 
directly from `Class` as opposed to the `Cosmology` class with the 
`classylss` package (which is also incompatible with my current setup). 
This appears to have led to a normalisation issue in the `CLASS` power spectrum model which appears to be 
the main bottleneck for the analysis at this stage. It should be straightforward, if time-consuming, 
to cross-check that `Class` is being used in the `LinearPower` and transfer 
function classes analogously to the `Cosmology` class implemented by `nbodykit`.
The implementation of the analytic models also needs to be cross-checked and refined.

## Acknowledgments
-----------------------------------------------------------------------
Many thanks to Dr. Florian Beutler (https://mystatisticsblog.blogspot.com), 
for the project pipeline and the original code upon which this project was 
based (without which this would have been a significantly longer and more 
arduous endeavour). Additional thanks to Dr. Mike Wang, for his `Triumvirate` 
package (https://github.com/MikeSWang/Triumvirate) and extensive assistance 
with this tool.
