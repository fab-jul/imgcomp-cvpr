from __future__ import print_function
import httplib2
import os
import argparse

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient.errors import HttpError


# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
APPLICATION_NAME = 'Sheets Logger'


class GoogleSheetsAccessFailedException(Exception):
    """ Generic exception thrown if something is not right """
    pass


# The following two ENVs must be set:
# SPREADSHEET_ID
# SHEETS_CLIENT_SECRET_FILE
def _get_spreadsheet_id():
    # '1pRy7ob8062QyD4roqOTqK5YIDXr2lX5-IKqcvYTmqLE')
    return _get_env('SPREADSHEET_ID',
                    info='ID of the google spreadsheet to store logs in.')


def _get_client_secret_file():
    return _get_env('SHEETS_CLIENT_SECRET_FILE',
                    info='Path to client_secret.json')


def _get_env(env_name, info):
    try:
        return os.environ[env_name]
    except KeyError:
        raise GoogleSheetsAccessFailedException(
                '*** Environment variable not set: {} ({})'.format(env_name, info))


# ------------------------------------------------------------------------------


def check_connection(flags=None):
    """ Checks if link to google sheet is correctly set up. """
    try:
        credentials = _get_credentials(flags)
        http = credentials.authorize(httplib2.Http())
        discovery_url = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
        service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discovery_url)

        title_cell = 'B2'
        title_cell_expected_content = 'Logs'

        result = service.spreadsheets().values().get(
            spreadsheetId=_get_spreadsheet_id(), range=title_cell).execute()
        values = result.get('values')
        if not values:
            raise GoogleSheetsAccessFailedException('No values found')
        if values[0][0] != title_cell_expected_content:
            raise GoogleSheetsAccessFailedException('Unexpected content found: {}'.format(values))
        print('Google Sheets connection established')
        return service
    except HttpError as e:
        raise GoogleSheetsAccessFailedException('HttpError: {}'.format(e))


def get_lock_file_p():
    client_secret_dir, client_secret_name = os.path.split(_get_client_secret_file())
    client_secret_name = os.path.splitext(client_secret_name)[0]  # remove ext
    spreadsheet_id = _get_spreadsheet_id()
    return os.path.join(client_secret_dir, '.{}_{}_lock'.format(client_secret_name, spreadsheet_id))


def insert_row(*row):
    service = check_connection()

    # Insert empty row
    body = {
        'requests': [{
            "insertDimension": {
                "range": {
                    "dimension": "ROWS",
                    "startIndex": 3,
                    "endIndex": 4
                },
            }
        }]
    }
    service.spreadsheets().batchUpdate(
            spreadsheetId=_get_spreadsheet_id(), body=body).execute()

    # Fill the new row
    body = {'values': [row]}
    result = service.spreadsheets().values().update(
            spreadsheetId=_get_spreadsheet_id(), range='Log!B4',
            valueInputOption='USER_ENTERED', body=body).execute()
    assert result['updatedCells'] == len(row)


def _get_credentials(flags):
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Params:
        flags: if given, create credentials with it. Run this script separatly for this.

    Returns:
        Credentials, the obtained credential.
    """
    client_secret_file = _get_client_secret_file()

    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-python-sheets_logger.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        if not flags:
            raise GoogleSheetsAccessFailedException(
                    'No credentials found. Run sheets_logger.py [--noauth_local_webserver].')
        flow = client.flow_from_clientsecrets(client_secret_file, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store, flags)
        print('Storing credentials to ' + credential_path)
    return credentials


def main():
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
    check_connection(flags)


if __name__ == '__main__':
    main()

