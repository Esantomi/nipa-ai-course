# 인공지능으로 주가 예측하기
이번 실습에서는 주식 데이터를 바탕으로 인공지능 모델을 사용하여 미래의 주식 가격을 예측하는 과정을 수행해 보겠습니다.

### 데이터 구조 확인
본격적으로 주식 가격을 예측해보기 전 **먼저 주식 데이터의 기본적인 구조 및 정보를 확인**해 보도록 하겠습니다.

데이터에는 **3월 2일부터 7월 30일까지 105개의 날짜별 주식 정보에 대한 값**들이 저장되어 있습니다.

![image](https://user-images.githubusercontent.com/61646760/145758015-3275f233-2e96-4daa-a3c0-e3875fc2edd5.png)

### 지시 사항
1. 아래 코드를 사용하여 출력된 데이터를 확인하고 학습을 수행해 보세요.
    - `ma.data_plot()`
2. 실행 버튼을 결과를 확인해 보세요.
3. 제출 버튼을 눌러 데이터 확인 및 학습을 올바르게 수행했는지 확인해 보세요.

## 답안
```
import machine as ma

def main():
    
    """
    지시사항 1번. 출력된 데이터를 확인하고 학습을 수행해 보세요.
    """
    ma.data_plot()

if __name__ == "__main__":
    main()
```

### 실행 결과
```
           Date   High    Low   Open  Close    Volume  Adj Close
0    2020-03-02  55500  53600  54300  55000  30403412      55000
1    2020-03-03  56900  55100  56700  55400  30330295      55400
2    2020-03-04  57600  54600  54800  57400  24765728      57400
3    2020-03-05  58000  56700  57600  57800  21698990      57800
4    2020-03-06  57200  56200  56500  56500  18716656      56500
5    2020-03-09  56500  56500  56500  56500         0      56500
6    2020-03-10  54900  53700  53800  54600  32106554      54600
7    2020-03-11  54400  52000  54300  52100  45707281      52100
8    2020-03-12  52100  52100  52100  52100         0      52100
9    2020-03-13  51600  46850  47450  49950  59462933      49950
10   2020-03-16  50900  48800  50100  48900  33339821      48900
11   2020-03-17  49650  46700  46900  47300  51218151      47300
12   2020-03-18  48350  45600  47750  45600  40152623      45600
13   2020-03-19  46650  42300  46400  42950  56925513      42950
14   2020-03-20  45500  43550  44150  45400  49730008      45400
15   2020-03-23  43550  42400  42600  42500  41701626      42500
16   2020-03-24  46950  43050  43850  46950  49801908      46950
17   2020-03-25  49600  47150  48950  48650  52735922      48650
18   2020-03-26  49300  47700  49000  47800  42185129      47800
19   2020-03-27  49700  46850  49600  48300  39896178      48300
20   2020-03-30  48350  46550  47050  47850  26797395      47850
21   2020-03-31  48500  47150  48000  47750  30654261      47750
22   2020-04-01  47900  45800  47450  45800  27259532      45800
23   2020-04-02  46850  45350  46200  46800  21621076      46800
24   2020-04-03  47600  46550  47400  47000  22784682      47000
25   2020-04-06  48800  47250  47500  48700  23395726      48700
26   2020-04-07  50200  49000  49650  49600  31524034      49600
27   2020-04-08  49750  48600  49600  48600  25010314      48600
28   2020-04-09  49800  48700  49750  49100  22628058      49100
29   2020-04-10  49250  48650  48950  49250  17839111      49250
..          ...    ...    ...    ...    ...       ...        ...
75   2020-06-19  52900  51600  52600  52900  18157985      52900
76   2020-06-22  52600  51800  52000  52000  13801350      52000
77   2020-06-23  52800  51100  52500  51400  18086152      51400
78   2020-06-24  53900  51600  51900  52900  24519552      52900
79   2020-06-25  53000  51900  52100  51900  18541624      51900
80   2020-06-26  53900  52200  52800  53300  21575360      53300
81   2020-06-29  53200  52000  52500  52400  17776925      52400
82   2020-06-30  53900  52800  53900  52800  21157172      52800
83   2020-07-01  53600  52400  53400  52600  16706143      52600
84   2020-07-02  52900  52100  52100  52900  14142583      52900
85   2020-07-03  53600  52700  53000  53600  11887868      53600
86   2020-07-06  55000  53800  54000  55000  19856623      55000
87   2020-07-07  55900  53400  55800  53400  30760032      53400
88   2020-07-08  53900  52900  53600  53000  19664652      53000
89   2020-07-09  53600  52800  53200  52800  17054850      52800
90   2020-07-10  53200  52300  53100  52700  13714746      52700
91   2020-07-13  53800  53100  53300  53400  12240188      53400
92   2020-07-14  53800  53200  53700  53800  14269484      53800
93   2020-07-15  55000  54300  54400  54700  24051450      54700
94   2020-07-16  54800  53800  54800  53800  16779127      53800
95   2020-07-17  54700  54100  54200  54400  10096174      54400
96   2020-07-20  54800  54000  54800  54200  10507530      54200
97   2020-07-21  55400  54800  55200  55300  18297260      55300
98   2020-07-22  55500  54700  55300  54700  12885057      54700
99   2020-07-23  54700  53800  54700  54100  16214932      54100
100  2020-07-24  54400  53700  54000  54200  10994535      54200
101  2020-07-27  55700  54300  54300  55600  21054421      55600
102  2020-07-28  58800  56400  57000  58600  48431566      58600
103  2020-07-29  60400  58600  60300  59000  36476611      59000
104  2020-07-30  60100  59000  59700  59000  19285354      59000

[105 rows x 7 columns]

주식 데이터의 형태를 출력
(105, 7)

주식 데이터의 정보를 출력
<bound method DataFrame.info of            Date   High    Low   Open  Close    Volume  Adj Close
0    2020-03-02  55500  53600  54300  55000  30403412      55000
1    2020-03-03  56900  55100  56700  55400  30330295      55400
2    2020-03-04  57600  54600  54800  57400  24765728      57400
3    2020-03-05  58000  56700  57600  57800  21698990      57800
4    2020-03-06  57200  56200  56500  56500  18716656      56500
5    2020-03-09  56500  56500  56500  56500         0      56500
6    2020-03-10  54900  53700  53800  54600  32106554      54600
7    2020-03-11  54400  52000  54300  52100  45707281      52100
8    2020-03-12  52100  52100  52100  52100         0      52100
9    2020-03-13  51600  46850  47450  49950  59462933      49950
10   2020-03-16  50900  48800  50100  48900  33339821      48900
11   2020-03-17  49650  46700  46900  47300  51218151      47300
12   2020-03-18  48350  45600  47750  45600  40152623      45600
13   2020-03-19  46650  42300  46400  42950  56925513      42950
14   2020-03-20  45500  43550  44150  45400  49730008      45400
15   2020-03-23  43550  42400  42600  42500  41701626      42500
16   2020-03-24  46950  43050  43850  46950  49801908      46950
17   2020-03-25  49600  47150  48950  48650  52735922      48650
18   2020-03-26  49300  47700  49000  47800  42185129      47800
19   2020-03-27  49700  46850  49600  48300  39896178      48300
20   2020-03-30  48350  46550  47050  47850  26797395      47850
21   2020-03-31  48500  47150  48000  47750  30654261      47750
22   2020-04-01  47900  45800  47450  45800  27259532      45800
23   2020-04-02  46850  45350  46200  46800  21621076      46800
24   2020-04-03  47600  46550  47400  47000  22784682      47000
25   2020-04-06  48800  47250  47500  48700  23395726      48700
26   2020-04-07  50200  49000  49650  49600  31524034      49600
27   2020-04-08  49750  48600  49600  48600  25010314      48600
28   2020-04-09  49800  48700  49750  49100  22628058      49100
29   2020-04-10  49250  48650  48950  49250  17839111      49250
..          ...    ...    ...    ...    ...       ...        ...
75   2020-06-19  52900  51600  52600  52900  18157985      52900
76   2020-06-22  52600  51800  52000  52000  13801350      52000
77   2020-06-23  52800  51100  52500  51400  18086152      51400
78   2020-06-24  53900  51600  51900  52900  24519552      52900
79   2020-06-25  53000  51900  52100  51900  18541624      51900
80   2020-06-26  53900  52200  52800  53300  21575360      53300
81   2020-06-29  53200  52000  52500  52400  17776925      52400
82   2020-06-30  53900  52800  53900  52800  21157172      52800
83   2020-07-01  53600  52400  53400  52600  16706143      52600
84   2020-07-02  52900  52100  52100  52900  14142583      52900
85   2020-07-03  53600  52700  53000  53600  11887868      53600
86   2020-07-06  55000  53800  54000  55000  19856623      55000
87   2020-07-07  55900  53400  55800  53400  30760032      53400
88   2020-07-08  53900  52900  53600  53000  19664652      53000
89   2020-07-09  53600  52800  53200  52800  17054850      52800
90   2020-07-10  53200  52300  53100  52700  13714746      52700
91   2020-07-13  53800  53100  53300  53400  12240188      53400
92   2020-07-14  53800  53200  53700  53800  14269484      53800
93   2020-07-15  55000  54300  54400  54700  24051450      54700
94   2020-07-16  54800  53800  54800  53800  16779127      53800
95   2020-07-17  54700  54100  54200  54400  10096174      54400
96   2020-07-20  54800  54000  54800  54200  10507530      54200
97   2020-07-21  55400  54800  55200  55300  18297260      55300
98   2020-07-22  55500  54700  55300  54700  12885057      54700
99   2020-07-23  54700  53800  54700  54100  16214932      54100
100  2020-07-24  54400  53700  54000  54200  10994535      54200
101  2020-07-27  55700  54300  54300  55600  21054421      55600
102  2020-07-28  58800  56400  57000  58600  48431566      58600
103  2020-07-29  60400  58600  60300  59000  36476611      59000
104  2020-07-30  60100  59000  59700  59000  19285354      59000

[105 rows x 7 columns]>

주식 ���이터의 상단 5개 행을 출력
         Date   High    Low   Open  Close    Volume  Adj Close
0  2020-03-02  55500  53600  54300  55000  30403412      55000
1  2020-03-03  56900  55100  56700  55400  30330295      55400
2  2020-03-04  57600  54600  54800  57400  24765728      57400
3  2020-03-05  58000  56700  57600  57800  21698990      57800
4  2020-03-06  57200  56200  56500  56500  18716656      56500

주식 데이터의 하단 5개 행을 출력
           Date   High    Low   Open  Close    Volume  Adj Close
100  2020-07-24  54400  53700  54000  54200  10994535      54200
101  2020-07-27  55700  54300  54300  55600  21054421      55600
102  2020-07-28  58800  56400  57000  58600  48431566      58600
103  2020-07-29  60400  58600  60300  59000  36476611      59000
104  2020-07-30  60100  59000  59700  59000  19285354      59000

주식 데이터의 모든 열을 출력
Index(['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], dtype='object')

주식 데이터의 요약 통계 자료 출력
               High           Low  ...        Volume     Adj Close
count    105.000000    105.000000  ...  1.050000e+02    105.000000
mean   51996.190476  50637.619048  ...  2.390007e+07  51310.476190
std     3278.145108   3365.597879  ...  1.152018e+07   3331.829995
min    43550.000000  42300.000000  ...  0.000000e+00  42500.000000
25%    49350.000000  48500.000000  ...  1.621493e+07  48800.000000
50%    51600.000000  50300.000000  ...  2.105442e+07  51200.000000
75%    54700.000000  53200.000000  ...  2.759696e+07  53800.000000
max    60400.000000  59000.000000  ...  5.946293e+07  59000.000000

[8 rows x 6 columns]
```