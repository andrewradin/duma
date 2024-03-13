# based on https://en.wikipedia.org/wiki/CAS_Registry_Number
def cas_error(s):
    l = list(reversed(s))
    if len(l) < 7:
        return 'too short'
    if l[1] != '-' or l[4] != '-':
        return 'misplaced hyphens'
    l = [x for i,x in enumerate(l) if i not in (1,4)]
    try:
        int(''.join(l))
    except ValueError:
        return 'non-digits'
    chk=[(i)*int(x) for i,x in enumerate(l)]
    if sum(chk)%10 != int(l[0]):
        return 'bad checksum'
    return None

