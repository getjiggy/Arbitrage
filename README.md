# Arbitrage

Main file to run is arb_driver.py

If running for the first time, answer "y" to "would you like to update tokens" and enter the amount of tokens you would like to check (Recommended 20 - 30 for a fast startup, but there is no limit). Tokens are pulled from the top traded tokens on the Pancakeswap API, so 20 would give you the 20 top tokens.

arb_driver.py will create the necessary route files for easier handling of the many token routes it generates. If running for a second time, answer "n" to the start-up prompt and it will use the previously generate routes.

Most of the output is printed to the console for now, but currently working on implementing some type of GUI or Dapp. Will print any profitable arbitrage trades with the token and factory addresses, but will only send to the chain if the profit covers gas fees.
