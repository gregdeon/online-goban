# ogs_auth.py
# Information used to connect to the Online Go client
# WARNING: don't commit sensitive information!

# OGS API client information
# Used to start OAuth2 session
# Obtained by making a client at https://online-go.com/oauth2/applications/
OGS_CLIENT_ID = 'ClientIDGoesHere012345678901234567890123'
OGS_CLIENT_SECRET = 'ClientSecretGoesHere012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567'

# OGS username and password
# Used to get authentication tokens for web API and real-time API
OGS_USERNAME = "your_username"
OGS_PASSWORD = "your_password"

# OGS user ID
# Can get this from browser while on online-go.com with console command:
# window.data.get('config.user').id
OGS_USER_ID = -1

# Authentication token for real-time API
# This seems to be undocumented; can get from browser with
# From window.data.get('config.chat_auth')
OGS_USER_AUTH = "abcdef01234567890123456789012345"
