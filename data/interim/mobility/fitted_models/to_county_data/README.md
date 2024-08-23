# Notes on results otained by fitting mobility models to the county-level data

## 2011-2015 commuter survey

- No mobility model (radiation basic/finite, dep.-diff. radiation, dep.-diff. powerlaw gravitation) was able to capture the survied mobility patterns at the US county-to-county level.

## 2016-2020 commuter survey

- Results and conclusion idem 2011-2015 data.

- Mobility is higher in 2016-2020 than in 2011-2015.

## SaveGraph data

- Radiation model: Clear trip density mismatch: modeled trip densities lower than measured trip densities. Poor fit to on-diagonal mobility.

- Departure-diffusion radiation model: Clear trip density mismatch: modeled trip densities lower than measured trip densities. Adequate fit to on-diagonal mobility.

- Departure-diffusion powerlaw gravitation model: Overlap between measured and modeled trip densities. R2=0.97. MAPE is quite high. Overall accurate. 

## Conclusion

The **commuter surveys should not be used to model US county-to-county level mobility patterns** as no model is able to adequately mimic the survied trip density. Neither should modeled county-to-county mobility patterns be aggregated to the US state level. Opposed, a departure-diffusion model with powerlaw gravitation calibrated to the cellphone (SafeGraph) data manages to mimic the measured density. 