class KnownIps:
    # to find a local IP address, use:
    # dig +short myip.opendns.com @resolver1.opendns.com
    #
    _known_ips = [
            # (key,ip,label)
            # key - if this IP address is used in a FirewallConfig rule, this
            #     will be in the form user.location (or, if the IP address
            #     belongs to one of our managed cloud machines, the machine
            #     name). For IP addresses not used by the firewall, this will
            #     be blank.
            # ip - dotted decimal IPv4 address
            # label - if key is not specified or is insufficiently descriptive,
            #     a label for this IP address to be shown when reporting
            #     logins or sessions
            ('friendly_name','192.168.1.20',"selenium"),
            ]
    @classmethod
    def hostname(cls,ip):
        for row in cls._known_ips:
            if ip == row[1]:
                return row[2] or row[0] or 'unlabeled '+row[1]
        return ip
    @classmethod
    def ssh_ips(cls,label):
        return [ip
                for fqn,ip,_ in cls._known_ips
                if label == fqn or label == fqn.split('.')[0]
                ]
