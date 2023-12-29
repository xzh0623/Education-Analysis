# 一、專案介紹
大多數人常常因為受到外在環境、家庭、教育等影響，造成每個人未來發展都有所不同。然而我們想探究哪些因素對於學生的正向發展可能產生極大的影響？
# 二、功能說明
從 Kaggle 中使用2019年工程學院與教育科學學院的學生資料集，分析哪些因素最能夠影響學生未來薪水所得？
## 資料特性
* 受測者數量共145人
* 特徵與在校學習狀況和家庭、環境因素相關，包含性別、上學的交通方式、每周學習時數、上課的出席情況、課堂筆記頻率、父母親教育程度等。共32筆
## 資料前處理
* 將資料分成numerical(包含15個特徵)和categorical(包含17個特徵)。
    * numerical data：使用Z-Score標準化，將資料介於[-3,3]
    * categorical data：使用One-Hot Encoding，將類別變數轉換為數值0/1
## 模型分析
* 隨機森林
* 線性回歸
* 類神經網路
# 三、結果
我們發現影響工程學院與教育科學學院的學生未來薪水所得，前5名較為相關的特徵如下：
* 學期GPA
* 父母親教育程度
* 高中就讀公立學校
* 領取獎學金面額
* 上課筆記習慣
# 四、遇到的困難
由於樣本數量太少，解釋範圍有限