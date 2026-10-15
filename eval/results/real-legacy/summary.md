| Group | Runs | First-attempt valid | Final valid | Expectations met | Task success | Pref. coverage | Submissions | Turns | Tokens in | Tokens out | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 30 | 87% | 100% | 90% | 90% | 93% | 1.1 | 6.7 | 27,243 | 3,990 | 42 |
| plan | 22 | 86% | 100% | 91% | 91% | 92% | 1.1 | 5.6 | 21,009 | 3,503 | 37 |
| replan | 8 | 88% | 100% | 88% | 88% | 94% | 1.1 | 9.8 | 44,385 | 5,329 | 56 |
| tag: baseline | 1 | 100% | 100% | 100% | 100% | 100% | 1.0 | 4.0 | 17,018 | 4,655 | 46 |
| tag: budget | 3 | 67% | 100% | 100% | 100% | 100% | 1.3 | 7.0 | 29,699 | 3,885 | 42 |
| tag: closures | 6 | 83% | 100% | 100% | 100% | 83% | 1.2 | 8.0 | 34,256 | 4,308 | 47 |
| tag: conflict | 1 | 0% | 100% | 100% | 100% | 0% | 2.0 | 10.0 | 45,018 | 4,740 | 53 |
| tag: delay | 1 | 100% | 100% | 100% | 100% | 100% | 1.0 | 3.0 | 10,288 | 4,335 | 36 |
| tag: dense | 4 | 75% | 100% | 75% | 75% | 95% | 1.2 | 5.5 | 19,907 | 3,428 | 36 |
| tag: distance | 4 | 100% | 100% | 50% | 50% | 88% | 1.0 | 6.8 | 31,305 | 4,820 | 51 |
| tag: evening | 1 | 100% | 100% | 100% | 100% | 100% | 1.0 | 8.0 | 31,440 | 3,179 | 38 |
| tag: museums | 1 | 100% | 100% | 100% | 100% | 100% | 1.0 | 4.0 | 10,067 | 2,833 | 30 |
| tag: preference-change | 3 | 100% | 100% | 67% | 67% | 83% | 1.0 | 11.7 | 47,222 | 4,943 | 56 |
| tag: replan | 8 | 88% | 100% | 88% | 88% | 94% | 1.1 | 9.8 | 44,385 | 5,329 | 56 |
| tag: short-window | 4 | 100% | 100% | 100% | 100% | 100% | 1.0 | 4.8 | 14,944 | 2,584 | 26 |
| tag: split-hours | 2 | 100% | 100% | 100% | 100% | 100% | 1.0 | 6.0 | 18,234 | 2,628 | 29 |
| tag: time-change | 1 | 100% | 100% | 100% | 100% | 100% | 1.0 | 9.0 | 40,554 | 4,601 | 49 |
| tag: travel-mode | 4 | 100% | 100% | 100% | 100% | 100% | 1.0 | 5.8 | 24,604 | 4,189 | 43 |
| tag: weather | 3 | 67% | 100% | 100% | 100% | 100% | 1.3 | 6.7 | 32,458 | 4,611 | 45 |

**First-attempt violation types:** insufficient_travel_time ×5

**Not successful:**
- `real-sat-walk-viking-fika` status=accepted expectations=['no stop of type viking'] violations=[]
- `real-sat-packed-day` status=accepted expectations=['no stop of type fika'] violations=[]
- `real-replan-add-viking` status=accepted expectations=['no stop of type viking'] violations=[]
