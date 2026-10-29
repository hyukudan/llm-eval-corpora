PYTHON ?= python3

.PHONY: verify verify-quick verify-full

verify-quick:   ## offline, seconds: regenerate the synthetic decision set
	$(PYTHON) verify.py --quick

verify:         ## download and rebuild every text dataset, check every sha256
	$(PYTHON) verify.py

verify-full:    ## + token-ID prompts and the MASSIVE / When2Call conversions
	$(PYTHON) verify.py --tokens --public
