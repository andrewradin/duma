from dtk.lazy_loader import LazyLoader
from dtk.subclass_registry import SubclassRegistry

class ApiKey(LazyLoader,SubclassRegistry):
    '''Base class for API key renewal.

    Class derived from this class each correspond to an API with a
    renewable key stored in the S3 keys bucket. This class supports
    derived class discovery, and reading and writing of the underlying
    file. The derived class must provide:
    - a 'name' member identifying this API to the user of the wrapper
      scripts/apikeys.py
    - a 'filename' member specifying the name of the file on S3 holding
      the key data
    - an 'expiry_date' member holding a python datetime.date indicating
      when a renewal needs to happen
    - a renew() member function that retrieves a new key
    The code under renew() may be broken out in a way that other code
    can share in order to access other API functions; see GDApiKey as
    an example.
    '''
    @classmethod
    def all_keys(cls):
        return [KeyType() for _,KeyType in cls.get_subclasses()]
    def _s3file_loader(self):
        from dtk.s3_cache import S3Bucket, S3File
        bucket = S3Bucket('keys')
        return S3File(bucket, self.filename)
    def _json_data_loader(self):
        self.s3file.fetch(force_refetch=True)
        import json
        with open(self.s3file.path(), 'r') as f:
            return json.loads(f.read())
    def rewrite(self,data):
        import json
        with open(self.s3file.path(), 'w') as f:
            json.dump(data,f)
        self.s3file.s3.bucket.put_file(self.filename)

class GDApiKey(ApiKey):
    name='GD'
    filename='global_data.json'
    def _expiry_date_loader(self):
        from datetime import date,datetime
        dt = datetime.strptime(self.json_data["ExpiryDate"],"%d/%m/%Y")
        return date(dt.year,dt.month,dt.day)
    def _api_key_loader(self):
        return self.json_data['TokenId']
    def api_fetch(self,endpoint,**kwargs):
        base_url = 'https://apidata.globaldata.com/GlobalDataAria'
        kwargs['api_key'] = self.api_key
        from django.utils.http import urlencode
        get_req=f'{base_url}/{endpoint}?{urlencode(kwargs)}'
        import requests
        rsp=requests.get(get_req)
        if rsp.status_code != 200:
            print(f'FETCH ERROR: status code == {rsp.status_code}')
            print(f'FAILING REQ: {get_req}')
            print(f'Returned body: {rsp.text}')
            print(f'Returned headers: {rsp.headers}')
            rsp.raise_for_status()
        return rsp.json()
    def renew(self):
        result = self.api_fetch('api/Token/TokenGeneration',
                OldTokenId=self.api_key,
                )
        # since getting a new key invalidates the old one, print the
        # response here so there's a chance of recovering if something
        # goes wrong with the write
        print(result)
        self.rewrite(result[0])

