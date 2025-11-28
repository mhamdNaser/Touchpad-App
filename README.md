
###########################################
##  Activate a virtual work environment  ##
###########################################

python -m venv venv
venv\Scripts\activate
uvicorn main:app --reload

python -m app.train_main train