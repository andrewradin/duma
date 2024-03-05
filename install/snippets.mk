default:
	false # no default target

mount:
	sudo mkdir -p /mnt2
	grep mnt2 /etc/fstab || sudo sh -c "sudo echo 'LABEL=DumaData	/mnt2	ext4	defaults	0	0' >> /etc/fstab"
	sudo mount /mnt2

splice_dvol:
	test "$(TO_MOVE)" != "" # TO_MOVE must be set
	test "$(DVOL)" != "" # DVOL must be set
	-mv $(TO_MOVE) $(TO_MOVE).old
	ln -s $(DVOL)/$(TO_MOVE) .

to_dvol:
	test "$(TO_MOVE)" != "" # TO_MOVE must be set
	test "$(DVOL)" != "" # DVOL must be set
	test \! -e $(DVOL)/$(TO_MOVE) # guard against overwrite
	cp -a $(TO_MOVE)/ $(DVOL)
	mv $(TO_MOVE) $(TO_MOVE).old
	ln -s $(DVOL)/$(TO_MOVE) .

