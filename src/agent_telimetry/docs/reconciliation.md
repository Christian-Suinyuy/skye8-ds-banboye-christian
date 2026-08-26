* catch hit only has effect on input tokens no effec on output tokens , Open Ai offers a 50% discount for cached request while anthropic offers a 90% discount. i cant seem to find the providers of models used in our system so in calculating actual cost i putt a placehoder for cache_discount and set it to 50%

* I believe models should not be billed when rate_limited as Open ai and nthropic do not charge for rate limitted calls, can be billed on timeout because the request might have gone through to the model and we just had a network or server failiure , and ofcourse billed for sucessfull calls. So some calles with same idempotent keys of timeout should still be billed.

* for some reason rate limited calls still have a completion token count wich should not be possible

* Billed rate limited calls = 423
* wrongly billed calls due to cache = 4565

* the all columns of the dataframe to 5 decimal places. the math cannot properly be accounted for becasue of the number of decimal places in the logged cost and the computed "Decimal" cost in 6 decimal places. so apparently all logged costs are frctionaly wrong