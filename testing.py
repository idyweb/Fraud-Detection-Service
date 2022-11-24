import requests

url = 'http://localhost:9696/fraud_detection_predict'

client = {"disrict"  :                  60,
"client_catg"  :                        11,
"region" :                             101,
"1transactions_count" :                 34,
"consommation_level_1_mean" :   442.735294,
"consommation_level_2_mean" :          0.0,
"consommation_level_3_mean"  :         0.0,
"consommation_level_4_mean":           0.0,
"year":                               2007,
"month":                                11 ,
"day" :                                 30,
"month_name":                     "November"

}

response = requests.post(url, json=client).json()
print(response)

if response['fraud'] == True:
    print('this client is fradulent.')
else:
    print('He is fine')





