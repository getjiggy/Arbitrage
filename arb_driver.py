import asyncio
import concurrent.futures
import json
import os
import sys
import time
import pickle
from decimal import Decimal
from math import sqrt
from pprint import pprint
from typing import Any
import numpy as np
from aiohttp import ClientSession
import greenlet
from web3.types import RPCEndpoint, RPCResponse
from eth_typing import URI
from requests import get
from itertools import combinations
from liq_pools import Pool, Route, duoTrade, decode_reserves
from web3 import Web3, HTTPProvider, exceptions
from collections import defaultdict
from aiolimiter import AsyncLimiter
from matplotlib import pyplot as plot


# wraps the normally synchronous make_request function with a greenlet
class AIOHTTPProvider(HTTPProvider):

    def make_request(self, method: RPCEndpoint, params: Any) -> RPCResponse:
        self.logger.debug("Making request HTTP. URI: %s, Method: %s",
                          self.endpoint_uri, method)

        request_data = self.encode_rpc_request(method, params)

        raw_response = self.green_await(self.make_post_request(
            self.endpoint_uri,
            request_data,
            **self.get_request_kwargs()
        ))

        response = self.decode_rpc_response(raw_response)
        self.logger.debug("Getting response HTTP. URI: %s, "
                          "Method: %s, Response: %s",
                          self.endpoint_uri, method, response)
        return response

    @staticmethod
    async def make_post_request(endpoint_uri: URI, data: bytes, *args: Any, **kwargs: Any) -> bytes:
        kwargs.setdefault('timeout', 10)
        try:
            async with ClientSession() as client:
                response = await client.post('https://bsc-dataseed.binance.org/', data=data, *args,
                                             **kwargs)  # type: ignore
                response.raise_for_status()
                return await response.content.read()
        except TimeoutError:
            pass

    @staticmethod
    def green_await(awaitable):
        current = greenlet.getcurrent()
        if not isinstance(current, AsyncIoGreenlet):
            raise TypeError('Cannot use green_await outside of green_spawn target')
        return current.driver.switch(awaitable)


class AsyncIoGreenlet(greenlet.greenlet):
    def __init__(self, driver, fn):
        greenlet.greenlet.__init__(self, fn, driver)
        self.driver = driver


class TriIntersect:

    @staticmethod
    def interpolated_intercepts(x, y1, y2):
        """Find the intercepts of two curves, given by the same x data"""

        def intercept(point1, point2, point3, point4):
            """find the intersection between two lines
            the first line is defined by the line between point1 and point2
            the first line is defined by the line between point3 and point4
            each point is an (x,y) tuple.

            So, for example, you can find the intersection between
            intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

            Returns: the intercept, in (x,y) format
            """

            def line(_p1, _p2):
                A = (_p1[1] - _p2[1])
                B = (_p2[0] - _p1[0])
                C = (_p1[0] * _p2[1] - _p2[0] * _p1[1])
                return A, B, -C

            def intersection(_L1, _L2):
                D = _L1[0] * _L2[1] - _L1[1] * _L2[0]
                Dx = _L1[2] * _L2[1] - _L1[1] * _L2[2]
                Dy = _L1[0] * _L2[2] - _L1[2] * _L2[0]
                _x = Dx / D
                _y = Dy / D
                return _x, _y

            L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
            L2 = line([point3[0], point3[1]], [point4[0], point4[1]])
            R = intersection(L1, L2)
            return R

        indices = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
        xcs = []
        ycs = []
        for idx in indices:
            xc, yc = intercept((x[idx], y1[idx]), (x[idx + 1], y1[idx + 1]), (x[idx], y2[idx]),
                               (x[idx + 1], y2[idx + 1]))
            xcs.append(xc)
            ycs.append(yc)
        return xcs, ycs

    # iteratively finds the intersect of the 3 liquidity pools price impact curves for optimal token amount to trade
    def find_tri_intersect(self, _route):
        p0_fee_adj = float(_route.pool0.fee_adj)
        p1_fee_adj = float(_route.pool1.fee_adj)
        p2_fee_adj = float(_route.pool2.fee_adj)

        p0_res_in, p0_res_out = _route.get_res_p0()
        p1_res_in, p1_res_out = _route.get_res_p1()
        p2_res_in, p2_res_out = _route.get_res_p2()
        res_list = [(p0_res_in, p0_res_out), (p1_res_in, p1_res_out), (p2_res_in, p2_res_out)]

        smallest = min(p0_res_in, p1_res_out, p2_res_out)
        x_values = np.linspace(1, np.float64(smallest-1e16), num=5000)
        y_values = x_values * p0_res_out * p0_fee_adj / p0_res_in + x_values * p0_fee_adj
        z_values = -(np.float64(_route.pool1.cp) / np.float64(p1_res_in + p1_fee_adj * y_values)) + p1_res_out
        with np.errstate(divide='raise'):
            right = (np.float64(_route.pool1.cp) / np.float64(p1_fee_adj * ((p1_res_out - z_values) ** 2))) * (
                    np.float64(_route.pool2.cp) / np.float64(p2_fee_adj * ((p2_res_out - x_values) ** 2)))
        left = np.float64(_route.pool0.cp * p0_fee_adj) / np.float64(((p0_fee_adj * x_values) + p0_res_in) ** 2)
        x_intersect, y_intersect = self.interpolated_intercepts(x_values, right, left)

        # Rounds down the token in amount result.
        if len(x_intersect) == 1:
            trunc_value = [int(x) for x in str(int(x_intersect[0][0]))]
            trunc_value[6:] = [0 for _ in trunc_value[6:]]
            final_value = int(''.join(map(str, trunc_value)))

            # uncomment to show plots for curve solutions
            # if final_value / 1e18 > 50 and _route.pool0_in in ['0x55d398326f99059ff775485246999027b3197955', '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56']:
                # print(final_value / 1e18, y_intersect[0][0], _route.pool0_in)
                # print(_route.pool0.tokens, _route.pool0.symbol, _route.pool0_in, _route.pool0_out)
                # print(_route.pool1.tokens, _route.pool1.symbol, _route.pool1_in, _route.pool1_out)
                # print(_route.pool2.tokens, _route.pool2.symbol, _route.pool2_in, _route.pool2_out)
                # plot.plot(x_intersect[0][0], y_intersect[0][0], '*k', ms=9)
                # plot.plot(x_values, left, 'b')
                # plot.plot(x_values, right, 'r')
                # plot.show()

            # if final_value / 1e18 > .3 and _route.pool0_in == '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c':
                # print(final_value / 1e18, y_intersect[0][0], _route.pool0_in)
                # print(_route.pool0.tokens, _route.pool0.symbol, _route.pool0_in, _route.pool0_out)
                # print(_route.pool1.tokens, _route.pool1.symbol, _route.pool1_in, _route.pool1_out)
                # print(_route.pool2.tokens, _route.pool2.symbol, _route.pool2_in, _route.pool2_out)
                # plot.plot(x_intersect[0][0], y_intersect[0][0], '*k', ms=9)
                # plot.plot(x_values[:2500], left[:2500], 'b')
                # plot.plot(x_values[:2500], right[:2500], 'r')
                # plot.show()

            return final_value, res_list
        else:
            return False, False


# Class using the async web3 client.
# Arb_driver class sends the updated pools from sync events here to perform the arb calculations and execute the trades.
# Arb_driver also uses this class to perform some start-up functions
class AsyncClient(AIOHTTPProvider):

    def __init__(self, sync_w3):
        super().__init__()
        self.aioprovider = AIOHTTPProvider(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
        self.web3_async = Web3(provider=self.aioprovider)
        self.web3_sync = sync_w3
        self.bnb = self.web3_async.toChecksumAddress('0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c')
        self.pkey = os.getenv('privateKey')
        self.tri_intersect = TriIntersect()

        self.pairs_dict = None
        self.cb = None
        self.keys = None
        self.tri_start_pools = None
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Duplicate storage here. Need to change so that arb_toks isn't saved twice.
        # Currently can't because we need arb tokens at startup, and startup needs an async client.
        self.arb_toks = {self.web3_async.toChecksumAddress('0x55d398326f99059ff775485246999027b3197955'),
                         self.web3_async.toChecksumAddress('0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56'),
                         self.web3_async.toChecksumAddress('0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c')}

    @staticmethod
    async def green_spawn(fn, *args, **kwargs):
        context = AsyncIoGreenlet(greenlet.getcurrent(), fn)

        result = context.switch(*args, **kwargs)

        while context:
            try:
                value = await result
            except Exception:
                result = context.throw(*sys.exc_info())
            else:
                result = context.switch(value)

        return result

    @staticmethod
    def shutdown():
        print("Shutting down")
        tasks = [t for t in asyncio.all_tasks()]
        for task in tasks:
            task.cancel()
        print('successfully shutdown')

    # Analytical solution for finding optimal token amount for duo token trades. Derived from AMM price impact formula.
    @staticmethod
    def find_intersect(a, b, a_res_in, b_res_out):
        init_value = (a.cp * b_res_out * a.fee_adj * b.fee_adj) + (a_res_in * b.cp * a.fee_adj)
        sqrt_value = Decimal((a.cp * b.cp * a.fee_adj * b.fee_adj) * ((b_res_out ** 2) * (
                a.fee_adj ** 2) + 2 * a_res_in * b_res_out * a.fee_adj + a_res_in ** 2)).sqrt()
        denominator = (a.fee_adj * (a.cp * b.fee_adj - b.cp * a.fee_adj))
        return int((init_value - sqrt_value) / denominator)

    # Below are functions taken from the pancakeswap and cake-like smart contracts. Used for calculating trades.
    @staticmethod
    def get_amount_out(amount_in, res_in, res_out, fee_adj):
        amount_in_with_fee = int(amount_in) * int(fee_adj * 10000)
        numerator = amount_in_with_fee * int(res_out)
        denominator = int(res_in * 10000) + amount_in_with_fee
        return int(numerator / denominator) - 10001

    @staticmethod
    def get_amount_in(amount_out, res_in, res_out, fee_adj):
        numerator = int(res_in * amount_out * 10000)
        denominator = int(res_out - amount_out) * int(fee_adj * 10000)
        return int(numerator / denominator) + 10001

    @staticmethod
    def k_check_out(pool, ai, ao, cp, fee_nominal, token_out):
        check = False

        if token_out == pool.token0:
            in_t = pool.token1
            out_t = pool.token0

            balance0 = int((pool.res0 - ao))
            balance1 = int(pool.res1 + ai)
            balance0_adj = int(balance0 * 10000)
            balance1_adj = int(balance1 * 10000) - int(ai * int(fee_nominal * 10000))

            left = balance0_adj * balance1_adj
            right = cp * (10000 ** 2)

            if left >= right:
                check = True

        else:
            in_t = pool.token0
            out_t = pool.token1

            balance0 = int((pool.res0 + ai))
            balance1 = int(pool.res1 - ao)
            balance0_adj = int(balance0 * 10000) - int(ai * int(fee_nominal * 10000))
            balance1_adj = int(balance1 * 10000)

            left = balance0_adj * balance1_adj
            right = cp * (10000 ** 2)

            if left >= right:
                check = True

        return check, ai / 1e18, ao / 1e18, in_t, out_t

    # Sends raw transaction to a factory contract to check if a token pair exists.
    def pair_address_from_factory(self, _fact_address, _pair0, _pair1):
        data_string = "0xe6a43905000000000000000000000000{token0}000000000000000000000000{token1}" \
            .format(token0=str(_pair0[2:]).lower(), token1=str(_pair1[2:]).lower())
        _raw_address = self.aioprovider.make_request('eth_call', ({'to': _fact_address, 'data': data_string}, 'latest'))
        return _raw_address

    # Function used to get the reserves when building liquidity pool objects.
    def get_reserves_initial(self, _contract_address):
        _raw_res = self.aioprovider.make_request('eth_call',
                                                 ({'to': _contract_address, 'data': '0x0902f1ac'}, 'latest'))
        _dec_res = decode_reserves(_raw_res['result'], self.web3_async)
        return _dec_res

    # main receiver function from arb_driver. Checks both duo and tri token routes built in arb_driver.
    # ptu: pairs to update, tsp: tri start pools
    def do_arb(self, _ptu, _tsp, _keys, _pairs_dict, _cb):
        st = time.time()
        self.pairs_dict = _pairs_dict
        self.cb = _cb
        self.keys = _keys
        self.tri_start_pools = _tsp
        try:
            print(f'getting res for {len(_ptu)} pools...')
            asyncio.run(self.get_res(_ptu))
            print(f'get res took {time.time() - st}')

            if len(self.tri_start_pools) > 0:
                self.check_tri_routes()

            ktime = time.time()
            asyncio.run(self.check_keys())
            print(f'duo check took {time.time() - ktime}')
        except asyncio.exceptions.CancelledError as ce:
            print(ce)
        print(f'-------------- {time.time() - st} --------------')

    # gets reserve amounts from the liquidity pools used to perform arb calculations.
    async def get_res(self, _pools):
        tasks = []
        for pool in _pools:
            tasks.append(self.green_spawn(self.get_reserves_tri, pool))
        try:
            await asyncio.gather(*tasks)
        except asyncio.exceptions.TimeoutError:
            print('Timeout:  Restarting arb checks in get_res...')
            self.shutdown()
        except asyncio.exceptions.CancelledError as ce:
            print(ce)
            self.shutdown()

    def get_reserves_tri(self, pool):
        pool.get_reserves(self.aioprovider, self.web3_async)

    # Main tri-route function. Incomplete.
    # Currently hands and pools in arb condition to the triIntersect class to find if any profitable trades are available.
    # TODO: add functionality to send/sign a profitable trade
    def check_tri_routes(self):

        tri_routes = []
        for pool in self.tri_start_pools:
            if len(pool.tri_routes) > 0:
                tri_routes[0:0] = pool.tri_routes
        print(len(tri_routes))
        condition_routes = [route for route in tri_routes if route.check_arb_conditions()]

        print(len(condition_routes))
        c_time = time.time()
        intersect_results = []
        for cr in condition_routes:
            final_value, res_list = self.tri_intersect.find_tri_intersect(cr)
            if final_value is not False:
                intersect_results.append((self.profit_conversion(cr.pool0_in, final_value), final_value, cr, res_list))

        if len(intersect_results) > 0:
            intersect_results = sorted(intersect_results, key=lambda tup: tup[0], reverse=True)

            for r in intersect_results:

                trade0_out = self.get_amount_out(r[1], r[3][0][0], r[3][0][1], r[2].pool0.fee_adj)
                check0 = self.k_check_out(r[2].pool0, r[1], trade0_out, r[2].pool0.cp, r[2].pool0.fee, r[2].pool0_out)

                trade1_out = self.get_amount_out(trade0_out, r[3][1][0], r[3][1][1], r[2].pool1.fee_adj)
                check1 = self.k_check_out(r[2].pool1, trade0_out, trade1_out, r[2].pool1.cp, r[2].pool1.fee, r[2].pool1_out)

                trade2_out = self.get_amount_out(trade1_out, r[3][2][0], r[3][2][1], r[2].pool2.fee_adj)
                check2 = self.k_check_out(r[2].pool2, trade1_out, trade2_out, r[2].pool2.cp, r[2].pool2.fee, r[2].pool2_out)

                if trade2_out > r[1]:
                    print(r[1], trade0_out, r[2].pool0_in, r[2].pool0_out, r[2].pool0.contract_address)
                    print(trade0_out, trade1_out, r[2].pool1_in, r[2].pool1_out, r[2].pool1.contract_address)
                    print(trade1_out, trade2_out, r[2].pool2_in, r[2].pool2_out, r[2].pool2.contract_address)
                    print(check0[0], check1[0], check2[0])
                    print('profit: ', (trade2_out - r[1]) / 1e18)
                    print('- - - - -')

        print(f'tri check took {time.time() - c_time}')

    # Task builder function for duo token trades. Gathers all potential pools by key/address pair.
    async def check_keys(self):
        _key_tasks = []
        for key in self.keys:
            _key_tasks.append(asyncio.create_task(self.duo_bid_ask_temp(key)))
        try:
            await asyncio.gather(*_key_tasks)
        except asyncio.exceptions.TimeoutError:
            print('Timeout:  Restarting arb checks in get_res...')
            self.shutdown()
        except asyncio.exceptions.CancelledError as ce:
            print(ce)
            self.shutdown()

    # Main function for duo arb trades. Compares a recently updated liquidity pool (sync event) to the
    # other available pools and checks if any are in arb condition. If any are profitable including all fees,
    # sends and signs the transaction.
    async def duo_bid_ask_temp(self, key):

        loop = asyncio.get_running_loop()
        arb_token = self.get_arb_token(key)

        # change for gas, don't forget liq_pools
        # currently set to low amount until personal node is established
        # TODO: add ability to dynamically change gas amount based on current block. Gas auctioning.
        gas = 5e9

        if arb_token is None:
            return

        sorted_pairs_sp, sorted_pairs_bp = self.price_sort(key, arb_token)
        if sorted_pairs_bp[-1].get_bp(arb_token) < sorted_pairs_sp[-1].get_sp(arb_token):
            for a, b in zip(sorted_pairs_bp, sorted_pairs_sp):
                if a.get_bp(arb_token) < b.get_sp(arb_token):
                    try:
                        p0, p1, p0_out_token, p1_out_token, high_amount, low_amount, check0, check1, x_value_neg = \
                            await self.organize_reserves(a, b, key, arb_token, loop)
                    except LookupError:
                        print('Started with BLP but no valid opposite token...', a.tokens)
                        return
                    except ValueError as ve:
                        print('X Value < 0', ve)
                        return
                    if check0[0] and check1[0]:
                        dif = high_amount - low_amount
                        if dif > 0:
                            in_out_t0 = [x_value_neg, high_amount]
                            in_out_t1 = [low_amount, x_value_neg]
                            trade = duoTrade(self.web3_async, p0, p1, in_out_t0, in_out_t1, p0_out_token)
                            ntime = time.time()
                            nonce = await self.green_spawn(self.get_nonce, trade.wallet)
                            print(f'nonce took {time.time() - ntime}')
                            try:
                                profit_to_bnb = self.profit_conversion(p0_out_token, dif)
                                est = 185000
                                print(profit_to_bnb / int(1e18), est * int(gas) / int(1e18), p0_out_token, p0.symbol,
                                      p1_out_token, p1.symbol)
                                if profit_to_bnb > est * gas:
                                    print(trade.swapDesc)
                                    print(arb_token)
                                    print(p0_out_token)
                                    btime = time.time()
                                    blk_check = await self.final_check(trade, nonce, self.cb)
                                    print(f'final check took {time.time() - btime}')
                                    print(blk_check)
                                    if blk_check[1]:
                                        print('*** arb ***')
                                        print(check0)
                                        print(check1)
                                        print(f'dif: {dif / 1e18}')
                                        print(f'profit check: {profit_to_bnb} dif, {est * gas} gas/chi cost')
                                        print((profit_to_bnb - int((est * gas / 1e18))), 'bnb profit')
                                        print(trade.arbTxn)
                                        # self.sendandsign(trade.arbTxn)
                                        input("Press Enter to continue...")
                            except KeyError as ke:
                                print(ke)
                                pass
                            except exceptions.ContractLogicError as cle:
                                print(cle)
                                pass
        print('----------')

    # Helper function for getting accurate nonce.
    def get_nonce(self, wallet):
        raw_request = self.aioprovider.make_request('eth_getTransactionCount', (wallet, 'latest'))
        return self.web3_async.toInt(hexstr=raw_request['result'])

    # Helper function for duo arbs. Makes sure the correct token is used.
    def get_arb_token(self, key):
        if self.bnb in key:
            return key.index(self.bnb)
        elif key[0] in self.arb_toks:
            return 0
        elif key[1] in self.arb_toks:
            return 1
        else:
            return None

    # Sorts the current price of given token key for use in the duo arb calculation. We want the most expensive token to
    # sell and the cheapest to buy.
    def price_sort(self, key, token):

        if token == 0:
            try:
                sorted_sell_list = sorted(self.pairs_dict[key], key=lambda P: P.token0_sp)
                sorted_buy_list = sorted(self.pairs_dict[key], key=lambda P: P.token0_bp, reverse=True)
                return sorted_sell_list, sorted_buy_list

            # used for debugging, shouldn't be necessary in the future but leaving just in case.
            except TypeError as te:
                print(f'*****{te}*****')
                for pool in self.pairs_dict[key]:
                    print(pool)
                    print(pool.token0_bp)
                    print(pool.token0_sp)
                    print(pool.token1_bp)
                    print(pool.token1_sp)
                    print(pool.res0)
                    print(pool.res1)
                    print('-----')

                return False

        elif token == 1:
            try:
                sorted_sell_list = sorted(self.pairs_dict[key], key=lambda P: P.token1_sp)
                sorted_buy_list = sorted(self.pairs_dict[key], key=lambda P: P.token1_bp, reverse=True)
                return sorted_sell_list, sorted_buy_list

            except TypeError as te:
                print(f'*****{te}*****')
                for pool in self.pairs_dict[key]:
                    print(pool)
                    print(pool.token0_bp)
                    print(pool.token0_sp)
                    print(pool.token1_bp)
                    print(pool.token1_sp)
                    print(pool.res0)
                    print(pool.res1)
                    print('-----')

                return False
        else:
            print('no match in sort_dict')
            return False, False

    # Helper function for duo arbs. Makes sure that the reserve amounts used are ordered properly for the arb calculations.
    async def organize_reserves(self, a, b, key, arb_token, loop):
        if a.symbol != 'BLP':
            a_res_in, a_res_out, a_out_token = a.get_res_bp(arb_token)
            b_res_in, b_res_out, b_out_token = b.get_res_sp(arb_token)
            x_value_neg = await loop.run_in_executor(None, self.find_intersect, a, b, a_res_in, b_res_out)
            if x_value_neg > 0:
                high_amount, low_amount, check0, check1 = self.get_trade_amounts(a, b, x_value_neg, a_res_in, a_res_out,
                                                                                 b_res_in, b_res_out,
                                                                                 a_out_token, b_out_token)
                return a, b, a_out_token, b_out_token, high_amount, low_amount, check0, check1, x_value_neg
            else:
                raise ValueError
        elif a.symbol == 'BLP' and key[abs(arb_token - 1)] in self.arb_toks:
            b_res_in, b_res_out, b_out_token = b.get_res_bp(abs(arb_token - 1))
            a_res_in, a_res_out, a_out_token = a.get_res_sp(abs(arb_token - 1))
            x_value_neg = await loop.run_in_executor(None, self.find_intersect, b, a, b_res_in, a_res_out)
            if x_value_neg > 0:
                high_amount, low_amount, check0, check1 = self.get_trade_amounts(b, a, x_value_neg, b_res_in, b_res_out,
                                                                                 a_res_in, a_res_out,
                                                                                 b_out_token, a_out_token)
                return b, a, b_out_token, a_out_token, high_amount, low_amount, check0, check1, x_value_neg
            else:
                raise ValueError
        else:
            raise LookupError

    # uses the standard cake-like functions to determine amounts to trade. Performs the kcheck to ensure that transaction
    # will not fail when sent to chain.
    def get_trade_amounts(self, a, b, x_value_neg, a_res_in, a_res_out, b_res_in, b_res_out, a_out_token, b_out_token):
        high_amount = self.get_amount_out(x_value_neg, a_res_in, a_res_out, a.fee_adj)
        low_amount = self.get_amount_in(x_value_neg, b_res_in, b_res_out, b.fee_adj)
        check0 = self.k_check_out(a, x_value_neg, high_amount, a.cp, a.fee, a_out_token)
        check1 = self.k_check_out(b, low_amount, x_value_neg, b.cp, b.fee, b_out_token)
        return high_amount, low_amount, check0, check1

    # finds the average price of a given token using any available pairs built in arb_driver. Proxy for an on chain oracle
    # used for converting gas fee given in BNB to whatever token is returned at the end of the arb. Makes for more accurate
    # profitablilty check.
    def profit_conversion(self, a_out_token, dif):
        if a_out_token == self.bnb:
            return int(dif)
        else:
            conv_list = self.pairs_dict[tuple(sorted([a_out_token, self.bnb]))]
            conv_token = conv_list[0].tokens.index(a_out_token)
            conv_list_avg = np.mean([pool.get_bp(conv_token) for pool in conv_list])
            return int(dif * conv_list_avg)

    # batches gas estimate, getting nonce, and the block check. Used before sending a transaction to ensure the accuracy
    # of the trade.
    async def final_check(self, trade, nonce, cb):
        tasks = [self.green_spawn(trade.gas_estimate, self.web3_async, nonce), self.green_spawn(self.block_check, cb)]
        try:
            return_array = await asyncio.gather(*tasks)
            return return_array
        except asyncio.exceptions.TimeoutError:
            print('Timeout:  Restarting arb checks in get_res...')
            self.shutdown()
        except asyncio.exceptions.CancelledError as ce:
            print(ce)
            self.shutdown()

    def block_check(self, cb):
        block = self.web3_async.eth.get_block_number()
        return block == cb

    def send_tx(self, tx):
        return self.web3_sync.eth.send_raw_transaction(tx.rawTransaction)

    def sign_tx(self, tx):
        tx = self.web3_sync.eth.account.sign_transaction(tx, private_key=self.pkey)
        return tx

    def sendandsign(self, arb_transaction):
        print('signing')
        signed = self.sign_tx(arb_transaction)
        print('----------------------signed-------------------------------')
        sent = self.send_tx(signed)
        print(signed)
        return sent


class ArbDriver:

    # build all the necessary components to start running the main loop.
    def __init__(self):
        start_t = time.time()
        self.provider = HTTPProvider("https://bsc-dataseed.binance.org/")
        self.w3 = Web3(self.provider)
        self.async_client = AsyncClient(self.w3)
        self.pcsPairAbi = None
        self.all_tokens = None
        self.token_pairs = None
        self.ex_pools = None
        self.all_pools = None
        self.pool_dict = None
        self.arb_toks = {self.w3.toChecksumAddress('0x55d398326f99059ff775485246999027b3197955'),
                         self.w3.toChecksumAddress('0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56'),
                         self.w3.toChecksumAddress('0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c')}
        # self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        print(self.w3.isConnected())
        get_tokens = str(input('Would you like to refresh the token list? (y/n): '))

        if get_tokens == 'y':
            self.token_requests_to_valid()
            self.get_duo_pools_from_valid()
        else:
            self.get_duo_pools_from_valid()

        if len(self.ex_pools) > 2:
            self.pair_dict = self.pair_dict_builder()
            self.sync_filter = self.build_filter()
            print(f'Successfully prepared filter.')
            print(f'Startup took: {time.time() - start_t}')
        else:
            raise ValueError(f"not enough pools in ex_pools, {len(self.ex_pools)} pools present.")

    # builds tri and duo routes by start pool for use in the main arb loop and appends to a file.
    @staticmethod
    def build_routes(_all_pools, arb_toks):

        pool_by_token = defaultdict(list)

        for _p in _all_pools:
            pool_by_token[_p.token0].append(_p)
            pool_by_token[_p.token1].append(_p)

        def get_tri_routes(lp_object, non_duo, _pool_by_token, _arb_token, _non_arb):
            _routes = []
            for pool in non_duo:
                try:
                    _idx = pool.tokens.index(_non_arb)
                    pool_start = pool.tokens[_idx]
                    pool_end = pool.tokens[abs(_idx - 1)]
                    step_pools = set(_pool_by_token[pool_end]) & set(_pool_by_token[_arb_token])
                    for sp in step_pools:
                        _routes.append([(_arb_token, _non_arb, lp_object.contract_address),
                                        (pool_start, pool_end, pool.contract_address),
                                        (pool_end, _arb_token, sp.contract_address)])
                except ValueError:
                    continue

            return _routes

        def route_builder(lp_object, _pool_by_token, _arb_token, _non_arb):
            tok0_routes = [p for p in filter(lambda x: x != lp_object, _pool_by_token[lp_object.token0])]
            tok1_routes = [p for p in filter(lambda x: x != lp_object, _pool_by_token[lp_object.token1])]
            all_tokens = [*tok0_routes, *tok1_routes]
            match = set(tok0_routes) & set(tok1_routes)
            non_duo = list(set(all_tokens).difference(match))
            lp_object.pools_in_routes = [lp_object, *list(match), *non_duo]
            return get_tri_routes(lp_object, non_duo, _pool_by_token, _arb_token, _non_arb)


        routes = defaultdict(list)
        for _pool in _all_pools:
            if '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c' in _pool.tokens:
                idx = _pool.tokens.index('0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c')
                routes[_pool.contract_address] = \
                    route_builder(_pool, pool_by_token, _pool.tokens[idx], _pool.tokens[abs(idx - 1)])
                continue
            elif _pool.token0 in arb_toks:
                routes[_pool.contract_address] = route_builder(_pool, pool_by_token, _pool.token0, _pool.token1)
                continue
            elif _pool.token1 in arb_toks:
                routes[_pool.contract_address] = route_builder(_pool, pool_by_token, _pool.token1, _pool.token0)
                continue
            else:
                continue

        with open('routes_by_pool.pickle', 'wb') as handle:
            pickle.dump(routes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    async def gather_with_concurrency(n, *tasks):
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    @staticmethod
    def get_cp(_res0, _res1):
        return _res0 * _res1

    # get all valid pools from a series of JSON RPC requests. Very expensive for large amounts of tokens.
    def token_requests_to_valid(self):
        token_number = int(input('How many tokens to check?: '))
        self.all_pools = self.getPairs(token_number)
        self.build_routes(self.all_pools, self.arb_toks)
        self.pcsPairAbi = None
        print(f'Gathered {len(self.all_pools)} pools.')

        _valid_pools = defaultdict(tuple)
        for pool in self.all_pools:
            _valid_pools[pool.contract_address] = (pool.token0, pool.token1, pool.symbol)

        # with open('valid_pools.pickle', 'wb') as handle:
        #     pickle.dump(_valid_pools, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('valid_pools.json', 'w') as f:
            json.dump(_valid_pools, f, indent=4)

    # First step of the json request method. Gets tokens from the pancakeswap api and gathers the factory contracts.
    # Initializes the event loop and passes in the factories to check.
    def getPairs(self, num_tokens):
        with open('pancake_fact_abi.json', 'r') as abi:
            pcsabi = json.load(abi)
        with open('cake_lp_abi.json', 'r') as abi:
            self.pcsPairAbi = json.load(abi)
        fd = {
            'Cake-v2': self.w3.eth.contract(self.w3.toChecksumAddress('0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'),
                                            abi=pcsabi),
            'Ape': self.w3.eth.contract(self.w3.toChecksumAddress('0x0841BD0B734E4F5853f0dD8d7Ea041c241fb0Da6'),
                                        abi=pcsabi),
            'Wault': self.w3.eth.contract(self.w3.toChecksumAddress('0xB42E3FE71b7E0673335b3331B3e1053BD9822570'),
                                          abi=pcsabi),
            'MDEX': self.w3.eth.contract(self.w3.toChecksumAddress('0x3CD1C46068dAEa5Ebb0d3f55F6915B10648062B8'),
                                         abi=pcsabi),
            'Biswap': self.w3.eth.contract(self.w3.toChecksumAddress('0x858E3312ed3A876947EA49d572A7C42DE08af7EE'),
                                           abi=pcsabi),
            'Bakery': self.w3.eth.contract(self.w3.toChecksumAddress('0x01bF7C66c6BD861915CdaaE475042d3c4BaE16A7'),
                                           abi=pcsabi),
        }
        self.all_tokens = self.get_adds(num_tokens)
        self.token_pairs = [tuple(sorted(tp, key=lambda combo: combo.upper())) for tp in
                            combinations(self.all_tokens, 2)]
        print(f'Checking {len(self.token_pairs)} pairs.')
        print('Gathering pools...')
        return asyncio.run(self.init_pools(fd))

    # Helper function for getPairs. Makes request to pancakeswap api for token info. Used for gathering top traded tokens
    # for use in route building.
    def get_adds(self, num_tokens):
        resp = get("https://api.pancakeswap.info/api/v2/tokens")
        decode = resp.json()
        token_list = []
        keys = list(decode['data'].keys())
        count = 0
        while count < num_tokens:
            token_list.append(self.w3.toChecksumAddress(keys[count]))
            count += 1
        return token_list

    # First step in the async loop.
    # Takes each factory and creates a task that will check if a pair address exists for each given pair.
    # Awaits the completion of all of the factory tasks.
    async def init_pools(self, _fd):
        print('Starting factory tasks...')
        start_loop = asyncio.get_running_loop()
        fact_tasks = []
        rate_limiter = AsyncLimiter(30, 1)
        for key, fact in _fd.items():
            print(f'Sending requests for {key}')
            fact_tasks.append(asyncio.create_task(self.fact_task(fact, start_loop, rate_limiter)))
        fact_results = await asyncio.gather(*fact_tasks)
        fact_list = list(filter(lambda x: x is not None, fact_results))
        final_pool_list = []
        for partial_list in fact_list:
            final_pool_list[0:0] = partial_list
        return final_pool_list

    # Second level of the async loop.
    # Takes a factory and creates a task for each possible pair in self.token_pairs.
    # Keeps track of how many requests are made for each factory.
    # Awaits all of the pair checks with concurrency. No more than N threads can be open per factory.
    async def fact_task(self, _fact, _loop, _limiter):
        pair_tasks = []
        start_time = time.time()
        test_requests = 0
        for pair in self.token_pairs:
            test_requests += 2
            pair_tasks.append(asyncio.create_task(self.pair_task(_fact.address, pair, _loop, _limiter)))
        pair_results = await self.gather_with_concurrency(50, *pair_tasks)
        return await _loop.run_in_executor(None, self.filter_pair_results, pair_results, _fact, start_time,
                                           test_requests)

    # Final level of the async loop.
    # Calls the factory to get a pair address, and if it exists, calls getReserves.
    # Request rate limiter is used here.  AsyncLimiter(X, Y) syntax: X = number of requests, Y = seconds
    # Example: AsyncLimiter(30, 1) means that no more than 30 requests can go out each second.
    # Currently set to 30/1 because BSC public rate limit is 10k/5min.  Should never hit the cap with this in place.
    async def pair_task(self, _fact_address, _pair, _loop, _limiter):
        async with _limiter:
            try:
                raw_address = await self.async_client.green_spawn(self.async_client.pair_address_from_factory,
                                                                  _fact_address, _pair[0], _pair[1])
                address = self.w3.toChecksumAddress('0x' + raw_address['result'][-40:])
                if address != '0x0000000000000000000000000000000000000000':
                    res = await self.async_client.green_spawn(self.async_client.get_reserves_initial, address)
                    if res[0] > 1e20 and res[1] > 1e20:
                        return address, _pair[0], _pair[1]
            except asyncio.TimeoutError:
                print(f'Timed out for {_pair} at {_fact_address}')
                return False, False, False

    # Builds pool object for each address that was successfully found.
    # Returns a list of pools for the given factory.
    def filter_pair_results(self, _pair_results, _fact, _start_time, _requests):
        address_list = list(filter(lambda x: x is not None, _pair_results))
        if len(address_list) > 0:
            symbol = self.w3.eth.contract(self.w3.toChecksumAddress(address_list[0][0]),
                                          abi=self.pcsPairAbi).functions.symbol().call()
            _pools_for_fact = [Pool(address, t0, t1, symbol) for address, t0, t1 in address_list if
                               address is not False]
            print(
                f'Got {len(_pools_for_fact)} pools for {symbol} in {time.time() - _start_time} seconds ({_requests} total requests)')
            return _pools_for_fact
        else:
            print(f'- No pools found for {_fact.address} -')
            return None

    # Lightweight option for starting client. Should be used most of the time.
    # Reads a dict of valid pools that was previously retrieved through token_requests_to_valid()
    # Turns the dict values to Pool objects and gets reserves for each object. Returns ex_pools.
    def get_duo_pools_from_valid(self):
        try:
            with open("valid_pools.json", 'r') as f:
                valid_pools = json.load(f)
                print(f'Checking {len(valid_pools)} pools...')
        except FileNotFoundError:
            input('Valid pools were never found. Do you want to check now?')
            self.token_requests_to_valid()
            with open("valid_pools.json", 'r') as f:
                valid_pools = json.load(f)
                print(f'Checking {len(valid_pools)} pools...')
        try:
            with open('routes_by_pool.pickle', 'rb') as handle:
                route_data = pickle.load(handle)
            self.all_pools, pools_w_arb_token, self.pool_dict = self.get_all_pools(_valid_pools=valid_pools)
            print(f'{len(pools_w_arb_token)} pools have arb token.')
            self.ex_pools = pools_w_arb_token
        except FileNotFoundError:
            print('Routes were never created. Making them now...')
            self.all_pools, pools_w_arb_token, self.pool_dict = self.get_all_pools(_valid_pools=valid_pools)
            self.build_routes(self.all_pools, self.arb_toks)
            with open('routes_by_pool.pickle', 'rb') as handle:
                route_data = pickle.load(handle)
            print(f'{len(pools_w_arb_token)} pools have arb token.')
            self.ex_pools = pools_w_arb_token

        for pool in self.ex_pools:
            pool.tri_routes = [Route((r[0][0], r[0][1]), self.pool_dict.get(r[0][2]),
                                     (r[1][0], r[1][1]), self.pool_dict.get(r[1][2]),
                                     (r[2][0], r[2][1]), self.pool_dict.get(r[2][2]))
                               for r in route_data[pool.contract_address]]

        print('Got Tri-Routes.')
        print('Getting reserves for all  pools...')
        asyncio.run(self.prep_pools(self.all_pools))
        print(f'Successfully got res for {len(self.all_pools)} pools.')

    def get_all_pools(self, _valid_pools):
        all_pools = []
        pools_w_arb_token = []
        pool_dict = defaultdict(list)
        for pool, value in _valid_pools.items():
            P = Pool(self.w3.toChecksumAddress(pool), self.w3.toChecksumAddress(value[0]),
                     self.w3.toChecksumAddress(value[1]), value[2])
            all_pools.append(P)
            pool_dict[pool] = P
            if value[0] in self.arb_toks or value[1] in self.arb_toks:
                pools_w_arb_token.append(P)
        return all_pools, pools_w_arb_token, pool_dict

    # Async driver for getting reserves in get_duo_pools_from_valid()
    async def prep_pools(self, _pools_to_prep):
        lp = asyncio.get_running_loop()
        res_tasks = []
        for p in _pools_to_prep:
            res_tasks.append(asyncio.create_task(self.get_reserves_async(p, lp)))
        return await asyncio.gather(*res_tasks)

    # Uses greenlets in async client to get reserves asynchronously.
    async def get_reserves_async(self, _p, _lp):
        reserves = await self.async_client.green_spawn(self.async_client.get_reserves_initial, _p.contract_address)
        if reserves[0] > 1e18:
            _p.res0 = int(reserves[0])
            _p.res1 = int(reserves[1])
            _p.cp = await _lp.run_in_executor(None, self.get_cp, _p.res0, _p.res1)
            await _lp.run_in_executor(None, _p.get_prices)
            return _p
        else:
            print(f'Reserves not past threshold for {_p.contract_address}.')
            print(input('Continue?:'))
            return _p

    # Turns ex_pools into two separate dicts.
    # _pair_dict = pools by pair, _pool_dict = key / value pair for contract_address / pool
    def pair_dict_builder(self):
        _pair_dict = defaultdict(list)
        for pool in self.ex_pools:
            _pair_dict[pool.tokens].append(pool)
        self.token_pairs = _pair_dict.keys()
        return _pair_dict

    # Adjusted web3 filter. Needed to get to the filter options directly.
    def build_filter(self):
        filter_options = {'topics': ['0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1'],
                          'address': [],
                          'fromBlock': 'latest',
                          'toBlock': 'latest'}
        for p in self.ex_pools:
            filter_options['address'].append(p.contract_address)
        return self.w3.eth.filter(filter_options)

    def get_block(self):
        return self.w3.eth.get_block_number()

    # ktc = keys to check
    def filter_keys(self, _sync_events):
        ktc = [p for p in
               set(list(p.tokens for p in [self.pool_dict.get(i) for i in set(list(i.address for i in _sync_events))]))
               if len(self.pair_dict.get(p)) > 1]
        return ktc

    def pools_to_update(self, _sync_events):
        return [self.pool_dict.get(i) for i in set(list(i.address for i in _sync_events))]

    # Main function to be run. Checks for emitted sync events when any liquidity pools are updated via a swap.
    # Passes any pools that have been updated to the do_arb() function in order to calculate if any arb opportunities
    # have been created by another trade. Only need to check pools that have been updated, improves run time.
    # TODO: add functionality to gracefully exit via input. Remove time.sleep() and improve flow for each block.
    def main(self):
        block_number = 0
        while True:
            cb = self.get_block()
            if cb != block_number:
                print(f'Current block is: {cb}')

                try:
                    sync_events = self.sync_filter.get_new_entries()

                    if len(sync_events) > 0:
                        fk_time = time.time()
                        keys_to_check = self.filter_keys(sync_events)
                        ptu = self.pools_to_update(sync_events)
                        tsp = [p for p in ptu if p in self.ex_pools]
                        print(f'filter keys time took {time.time() - fk_time}, checking {len(keys_to_check)} keys.')
                        self.async_client.do_arb(ptu, tsp, keys_to_check, self.pair_dict, cb)
                        block_number = cb
                    else:
                        block_number = cb
                        continue

                except ValueError:
                    print('filter not found probably...')
                    self.sync_filter = self.build_filter()
                    block_number = cb

            else:
                time.sleep(.25)


if __name__ == '__main__':
    arb_driver = ArbDriver()
    arb_driver.main()
