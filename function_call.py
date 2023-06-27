from dotenv import load_dotenv
import openai
from testnew import send
import json
import os
import json
import sib_api_v3_sdk as SibApiV3Sdk

def send(body):
# Initialiser la clé API
  configuration = SibApiV3Sdk.Configuration()
  configuration.api_key['api-key'] = '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'

  # Initialiser l'API client
  api_instance = SibApiV3Sdk.TransactionalEmailsApi(SibApiV3Sdk.ApiClient(configuration))

  # Définir les destinataires de l'e-mail :email="dg@restoconcept.com";"france@restoconcept.com"
  to = [SibApiV3Sdk.SendSmtpEmailTo(email="france@restoconcept.com", name="becem")]

  # Définir l'expéditeur de l'e-mail
  sender = SibApiV3Sdk.SendSmtpEmailSender(email="chat@restoconcept.com", name="Van Brucken Group")

  # Définir les paramètres de l'e-mail
  subject = "information de produit"

  params = SibApiV3Sdk.SendSmtpEmail(
      to=to,
      sender=sender,
      subject=subject,
      # html_content=html_content,
     html_content="<p>"+body+"/<p>"
  )


 
  # Envoyer l'e-mail
  try:
      # Appeler l'API pour envoyer l'e-mail
      api_response = api_instance.send_transac_email(params)
      print("E-mail envoyé avec succès à {0} destinataire(s)".format(len(to)))
  except SibApiV3Sdk.ApiException as e:
      print("Exception lors de l'envoi de l'e-mail: %s\n" % e)

load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")
content= "  quel est le prix de vitrine esasy 900 "
# content="bonjour "
def mail(c):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role":"user","content":c}],
        functions=[
            {
                "name":"send",
                "description":"envoyer un eamil lorsque le client demand le prix d'un produit ",
                "parameters": {
                    "type" : "object",
                    "properties": {
                        "body":{
                            "type":"string",
                            "description":"content of the email",
                        },
                    },
                    "required": ["body"],
                },
            }
        
        ],
        function_call="auto",
    ) 
    message =completion.choices[0]['message']
    if message.get('function_call'):
        name=message['function_call']["name"]
        body=eval(message['function_call']["arguments"]).get("body")
        send(body) 
        print(name)

# content= "send an email to check the price of product "
# data = draft_email(content=content)
# print (data)
# send(data['body'])