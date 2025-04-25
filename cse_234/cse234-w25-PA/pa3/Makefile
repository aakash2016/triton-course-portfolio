FILES= part1/ part2/ part3/

.PHONY: check clean all

# Make handin.tar the default target
all: handin.tar

check:
	@chmod +x check_submission.sh
	@./check_submission.sh

handin.tar: check $(FILES)
	tar cvf handin.tar --exclude="*.DS_Store" $(FILES)
	@echo "handin.tar is ready."

clean:
	rm -f *~ handin.tar