# S3S4-detection
source:big_exam
#:634

| method  | channel  | time | val_loss | test_acc |
| wavelet | hs1+hs2  | 0-10s| 0.6148   | 0.6492 |
| HHT      | hs1+hs2 | 2-8s | 0.603    | 0.6190 |
| HHT      | hs1+hs2 | 0-10s | 0.6037  | 0.6666 |
| HHT | hs1(IMF0-2)  |    0-10s | 0.5997 | 0.6282|
| HHT(only frequency) | hs1(IMF0-2)  |    0-10s | 0.6014 | 0.5916|
| EMD | hs1(IMF0-2)  |    0-10s | 0.6613 | 0.5130|