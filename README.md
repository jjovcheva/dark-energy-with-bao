# Constraining Dark Energy with Baryon Acoustic Oscillations (v2)

## Context
-----------------------------------------------------------------------
This is a reprise of a project I completed in my third year of university.
I intended to only briefly revisit it, but the original directory was a
mess, I had moved from using `nbodykit` (which no longer runs on my 
laptop) to `Triumvirate` for power-/bispectrum estimation, and after another
year of cosmology I was much better equipped to appreciate and execute
it -- so I decided to redo it. The final constraint from the original project
can be found in `plots`.

## Current Issues
-----------------------------------------------------------------------
The models for the power spectrum are surprisingly tricky to implement
without `nbodykit`. I have moved to using `Colossus`, but there appear to be
normalisation issues with this approach. 

## Acknowledgments
-----------------------------------------------------------------------
Many thanks to Dr. Florian Beutler (https://mystatisticsblog.blogspot.com), 
for the project pipeline and the original code upon which this project was 
based (without which this would have been a significantly longer and more 
arduous endeavour). Additional thanks to Dr. Mike Wang, for his `Triumvirate` 
package (https://github.com/MikeSWang/Triumvirate) and extensive assistance 
with this tool.
