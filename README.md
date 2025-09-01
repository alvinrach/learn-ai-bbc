# learn-ai-bbc
Kaggle dataset https://www.kaggle.com/c/learn-ai-bbc/

## API

Installation

```
python3.12 -m venv .venv
. .venv/bin/activate
```

then

```
pip install uvicorn fastapi torch transformers nltk contractions
```

or

```
pip install -r requirements.txt
```

Start deployment by

```
uvicorn main:app
```

Sample Input

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "tsunami hit sri lanka banks sri lanka banks face hard times following december tsunami disaster officials warned sri lanka banks association said waves killed people also washed away huge amounts property securing loans according estimate much loans made private banks clients disaster zone written damaged state owned lenders may even worse hit said association estimates private banking sector bn rupees loans outstanding disaster zone one hand banks dealing death customers along damaged destroyed collateral extending cheap loans rebuilding recovery well giving clients time repay existing borrowing combination means revenue shortfall slba chairman commercial bank managing director al gooneratne told news conference banks given moratoriums collecting interest least quarter said public sector one ten state owned people bank customers south sri lanka affected bank spokesman told reuters estimated bank loss bn rupees",
    "career honour actor dicaprio actor leonardo dicaprio exceptional career honoured santa barbara international film festival star presented award martin scorsese directed oscar nominated movie aviator lifetime achievement award completely utterly surreal given years old dicaprio said almost years done quite films retrospective movies shown really exciting really love added want rest life dicaprio began movie career horror film critters moving onto roles basketball diaries romeo juliet titanic gangs new york achievement award created commemorate california festival th anniversary coincided dicaprio portrayal millionaire howard hughes aviator veteran actress jane russell starred hughes film outlaw said impressed dicaprio quest authenticity previously discussed role happy dicaprio came cared come find hughes really like said aviator taken pole position year oscars race nominations including nominations best film best actor dicaprio best director scorsese"
  ]
}'
```
Result

```
0 business	
1 entertainment	
2 politics	
3 sport	
4 tech
```

Able to do batch prediction