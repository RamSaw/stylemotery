
import urllib.request
import smtplib

def send_email(user, pwd, recipient, subject, body):
    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")

keyword = "Tickets for this event are not yet on sale."
url = "http://libraryphila.tix.com/Event.aspx?EventCode=897484"


opener = urllib.request.FancyURLopener({})
f = opener.open(url)
content = f.read()
if keyword in content.decode("utf-8"):
    send_email("bander@alsulamy.com","Zen@C00l","bander@alsulamy.com","Yes","YES YES")
else:
    print("no")