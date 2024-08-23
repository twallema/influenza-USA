# Notes on tested mobility models

## 2011-2015 commuter survey

The output of the basic and finite radiation models did not differ, so finite radiation was not considered further. These models specifically overestimate the number of people traveling from Hawaii to Alaska. However, we recommend having a look at the number of commutes originating from Alaksa (02) and Hawaii (15) in general, as there are several absurdities in the prediction. The accuracy of the radiation models is quite low with an R2 = 0.63. The departure-diffusion model with radiation diffusion is more realistic because it hierarchally estimates the fraction of the population leaving the origin and then partitions the commuters that leave in accordance with a radiation model, it has an R2=0.89 on the dataset. Gravity models were not tested because these are generally recognized as providing less adequate results in general.

## 2015-2016 commuter survey

Results for the radiation models are are again quite low with an R2 = 0.63. The difference with the 2011-2015 data seems very small. The departure-diffusion radiation model has an R2 = 0.89, mobility seems higher in 2016-2020 than in 2011-2015.

## SaveGraph data

The departure-diffusion radiation model achieves an R2 = 0.95 while the radiation model has an R2 = 0.60. By comparing the histograms of the data and the model it seems the dep.-diff. rad. model provides a good estimate of on-diagonal trips, as well as good estimates for off-diagonal flows with a size down to about 5%-10% of a counties population. Smaller flows are systematically underpredicted. For a radiation model the on-diagonal trips are modeled less accurately, but off-diagonal trips down to 1% are adequately modeled. Smaller flows are systematically underpredicted.

However, perhaps it is wrong to compare the dataset and the model so directly. Closer analysis of the dataset clearly reveals some caveats, for instance, 113 counties from all over the US have flows into Loving county TX (48301), the US's smallest county with a population of 64 (sum: 6262). Some counties might have disproportionately high or low counts due to local events, seasonal variations, or sample anomalies.