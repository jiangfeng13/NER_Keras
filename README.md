# NER_Keras

###using keras + tensorflow for NER###

requirement：tensorflow == 1.14  ,keras = 2.2.0, keras-self-attention,keras_multi_head

##Model

embedding + multiheadattention + BiLSTM + CRF


## Data Set
中	O
华	O
人	O
民	O
共	O
和	O
国	O

data from MSRA

you can use yourself's  data, only need to change the code in  `  self_data `

#HOW to use

`python run.py --mode=train `

## Reference
https://github.com/stephen-v/zh-NER-keras
