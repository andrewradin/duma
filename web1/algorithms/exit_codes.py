from path_helper import PathHelper
from collections import namedtuple

def status_file_line(code):
    return "FAILED: return code=%d\n" % code

class ExitCoder:
    '''Convert between numeric error codes and shared meanings.

    Errors are listed in the exit_codes.tsv file.  At the point of error
    detection, a symbolic name can be translated to a numeric code.  At
    the point of error handling, a numeric code can be tested to see if
    it matches a symbol.  This interface allows us to catch typos in
    symbol names on both ends, and potentially to allow multiple symbols
    to match to the same code. (The inverse doesn't make sense, because
    you need a single code for a symbol at the point of error detection.)

    Note that if we do map multiple symbolic errors to the same numeric
    code, message_of_code() will return the first match in the file.

    The tsv file can be easily parsed by both python and R.  Potentially,
    a third column could be added to the tsv file to hold a detailed
    error explanation to be presented to a user.
    '''
    CodeType = namedtuple('CodeType','code symbol message')
    def __init__(self):
        self.code_list = []
        with open(PathHelper.exit_codes) as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                self.code_list.append(self.CodeType(
                        int(fields[0]),
                        fields[1],
                        fields[2] if len(fields) > 2 else ''
                        ))
    def encode(self,symbol):
        for row in self.code_list:
            if row.symbol == symbol:
                return row.code
        raise Exception("lookup of unconfigured error symbol '%s'" % symbol)
    def decodes_to(self,symbol,code):
        symbol_seen = False
        for row in self.code_list:
            if row.symbol == symbol:
                if row.code == int(code):
                    return True
                else:
                    symbol_seen = True
        if symbol_seen:
            return False
        raise Exception("lookup of unconfigured error symbol '%s'" % symbol)
    def message_of_code(self,code):
        for row in self.code_list:
            if row.code == int(code):
                if row.message:
                    return row.message
                return "%s (%d)" % (row.symbol,row.code)
        return "exit code %d" % int(code)

