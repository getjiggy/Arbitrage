from decimal import Decimal
from web3 import Web3
import json

bnb = Web3.toChecksumAddress('0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c')

with open('duoContractAbi.json', 'r') as duo:
    duo_abi = json.load(duo)

fee_dict = {'Cake-LP': Decimal(".0025"),
            'APE-LP': Decimal(".002"),
            'MDEX LP': Decimal(".003"),
            'Cheese-LP': Decimal(".002"),
            'WLP': Decimal(".002"),
            'BSW-LP': Decimal(".001"),
            'TLP': Decimal(".004"),
            'BLP': Decimal(".003"),
            'SLP': Decimal(".003")
            }


def decode_reserves(result, web3):
    reserves = []
    for i in range(2, len(result), 64):
        reserves.append(web3.toInt(hexstr=result[i:i + 64]))
    return reserves


def encode_calc(uints):
    methodid = 'a95b089f'
    data = Web3.solidityKeccak(['uint8', 'uint8', 'uint256'], [uints[0], uints[1], uints[2]])
    _hex = Web3.toHex(data)
    _str = '0x' + methodid + _hex[2:]
    return _str


def encode_index(i):
    method_id = '91ceb3eb'
    hex_i = Web3.toHex(i)
    string = '0x' + method_id + '000000000000000000000000000000000000000000000000000000000000000' + hex_i[2:]
    return string


def decode_a(result, web3):
    a = web3.toInt(hexstr=result[2:])

    return a


def decode_bal(result, web3):
    bal = web3.toInt(hexstr=result[2:])

    return bal


class Pool:

    def __init__(self, _address, _token0, _token1, _symbol):

        self.symbol = _symbol
        self.fee = fee_dict[self.symbol]
        self.fee_adj = 1 - self.fee
        self.contract_address = _address
        self.token0 = _token0
        self.token1 = _token1
        self.tokens = (self.token0, self.token1)
        self.res0 = None
        self.res1 = None
        self.token0_price = None
        self.token1_price = None
        self.cp = None
        self.liq = None
        self.token0_bp = None
        self.token0_sp = None
        self.token1_bp = None
        self.token1_sp = None
        self.tri_routes = None

    def get_reserves(self, provider, web3):
        try:
            _raw_res = provider.make_request('eth_call', ({'to': self.contract_address, 'data': '0x0902f1ac'}, 'latest'))
            _dec_res = decode_reserves(_raw_res['result'], web3)
            self.res0 = int(_dec_res[0])
            self.res1 = int(_dec_res[1])
            self.cp = int(self.res0 * self.res1)
            self.get_prices()  # move to a more efficient place eventually

            if self.token0_sp is None:
                print('stupid error')
                pass

            return _dec_res

        except TypeError:
            print(f'get res failed for {self.contract_address, self.symbol, self.tokens}')

    def get_prices(self):
        self.token0_price = Decimal(self.res1) / Decimal(self.res0)
        self.token1_price = Decimal(self.res0) / Decimal(self.res1)
        self.token0_bp = self.token0_price / self.fee_adj
        self.token0_sp = self.token0_price * self.fee_adj
        self.token1_bp = self.token1_price / self.fee_adj
        self.token1_sp = self.token1_price * self.fee_adj

    def get_price_impact(self):
        self.liq = Decimal(self.cp).sqrt()
        # pi = self.cp / Decimal((self.res0 - 1) ** 2)
        # self.pi_percent = (pi - self.token0_price) / self.token0_price

    def get_bp(self, token_number):
        if token_number == 0:
            return self.token0_bp
        elif token_number == 1:
            return self.token1_bp
        else:
            raise ValueError

    def get_bp_token(self, _token):
        if _token == self.token0:
            return self.token0_bp
        elif _token == self.token1:
            return self.token1_bp
        else:
            raise ValueError

    def get_sp(self, token_number):
        if token_number == 0:
            return self.token0_sp
        elif token_number == 1:
            return self.token1_sp
        else:
            raise ValueError

    def get_sp_token(self, _token):
        if _token == self.token0:
            return self.token0_sp
        elif _token == self.token1:
            return self.token1_sp
        else:
            raise ValueError

    def get_res_out_token(self, _out_token):
        if _out_token == self.token0:
            return self.res1, self.res0
        elif _out_token == self.token1:
            return self.res0, self.res1
        else:
            raise ValueError

    def get_res_bp(self, token_number):
        if token_number == 0:
            return self.res1, self.res0, self.token0
        elif token_number == 1:
            return self.res0, self.res1, self.token1
        else:
            raise ValueError

    def get_res_sp(self, token_number):
        if token_number == 0:
            return self.res0, self.res1, self.token1
        elif token_number == 1:
            return self.res1, self.res0, self.token0
        else:
            raise ValueError

    def __gt__(self, other):
        return self.token0_sp > other.token0_bp

    def __lt__(self, other):
        return self.token0_bp < other.token0_sp


class Route:

    def __init__(self, pool0_order: tuple, _pool0: Pool, pool1_order: tuple, _pool1: Pool, pool2_order: tuple, _pool2: Pool):

        self.pool0 = _pool0
        self.pool1 = _pool1
        self.pool2 = _pool2

        self.pool0_in = pool0_order[0]
        self.pool1_in = pool1_order[0]
        self.pool2_in = pool2_order[0]
        self.pool0_out = pool0_order[1]
        self.pool1_out = pool1_order[1]
        self.pool2_out = pool2_order[1]

    def check_arb_conditions(self):
        return self.pool0.get_sp_token(self.pool0_in) > self.pool1.get_bp_token(self.pool1_out) * self.pool2.get_bp_token(self.pool2_out)

    def get_res_p0(self):
        if self.pool0_in == self.pool0.token0:
            return self.pool0.res0, self.pool0.res1
        elif self.pool0_in == self.pool0.token1:
            return self.pool0.res1, self.pool0.res0
        else:
            raise ValueError

    def get_res_p1(self):
        if self.pool1_out == self.pool1.token0:
            return self.pool1.res1, self.pool1.res0
        elif self.pool1_out == self.pool1.token1:
            return self.pool1.res0, self.pool1.res1
        else:
            raise ValueError

    def get_res_p2(self):
        if self.pool2_out == self.pool2.token0:
            return self.pool2.res1, self.pool2.res0
        elif self.pool2_out == self.pool2.token1:
            return self.pool2.res0, self.pool2.res1
        else:
            raise ValueError


class StablePool:
    def __init__(self, contract, symbol):
        self.contract = contract
        self.contract_address = self.contract.address
        self.symbol = symbol
        self.token_list = []
        self.index = []
        self.flag = 1
        self.balances = []

    def construct_token_list(self):

        for i in range(5):
            try:
                token = self.contract.functions.getToken(i).call()
                if token != '0x0000000000000000000000000000000000000000':
                    self.token_list.append(token)
                    self.index.append(i)
            except:
                pass

    def get_a(self, provider, web3):
        _raw_a = provider.make_request('eth_call', ({'to': self.contract_address, 'data': '0x0ba81959'}, 'latest'))
        _dec_a = decode_a(_raw_a['result'], web3)
        self.a = _dec_a

    def get_reserves(self, provider, web3, i):
        data = encode_index(i)
        _raw_bal = provider.make_request('eth_call', ({'to': self.contract_address, 'data': data}, 'latest'))
        _dec_bal = decode_bal(_raw_bal['result'], web3)
        self.balances.append(_dec_bal)

    def get_index(self, in_add, out_add):
        index = [self.token_list.index(in_add), self.token_list.index(out_add)]
        return index


class duoTrade:
    def __init__(self, w3, t0_pool, t1_pool, t0_in_out, t1_in_out, arb_token):
        self.duo_contract_address = w3.toChecksumAddress('0xDB8Aaa18D459D7D63066adD70D5c484dF53c11E5')
        self.duo_contract = w3.eth.contract(address=self.duo_contract_address, abi=duo_abi)
        self.wallet = w3.toChecksumAddress('0xAF2C3338BCaAACd8BcB9102B32ba4f936A733EB0')
        self.t0_amts = t0_in_out
        self.t1_amts = t1_in_out
        self.t0_pool = t0_pool
        self.t1_pool = t1_pool
        self.arb_token = arb_token
        self.egas = None
        self.arbTxn = None

        if self.t0_pool.token0 == self.arb_token:
            # print('t0')
            # print(self.t0_pool.contract_address, self.t0_pool.symbol, self.t1_pool.contract_address, self.t1_pool.symbol)
            self.swapDesc = [int(self.t0_pool.fee * 10000), int(self.t1_pool.fee * 10000), self.t0_amts[1], 0,
                            [t0_pool.contract_address, t1_pool.contract_address, self.arb_token]]
        elif self.t0_pool.token1 == self.arb_token:
            # print('t1')
            # print(self.t0_pool.contract_address, self.t0_pool.symbol, self.t1_pool.contract_address,
            #       self.t1_pool.symbol)
            self.swapDesc = [int(self.t0_pool.fee * 10000), int(self.t1_pool.fee * 10000), 0, self.t0_amts[1],
                             [t0_pool.contract_address, t1_pool.contract_address, self.arb_token]]
            # self.swapDesc = [w3.toInt(hexstr=self.t0_pool.contract_address), w3.toInt(hexstr=self.t0_pool.token0),
            #                  self.t0_amts[0], 0,
            #                  self.t0_amts[1], int(self.t0_pool.fee * 10000),
            #                  w3.toInt(hexstr=self.t1_pool.contract_address), w3.toInt(hexstr=self.t1_pool.token1),
            #                  self.t1_amts[0],
            #                  self.t1_amts[1], 0, int(self.t1_pool.fee * 10000)]

    def gas_estimate(self, web3, nonce):

        arbTxn = self.duo_contract.functions._duoSwap(self.swapDesc[0], self.swapDesc[1], self.swapDesc[2], self.swapDesc[3],
                                                      self.swapDesc[4]).buildTransaction({
            'from': self.wallet,
            'chainId': 56,
            'gas': int(300000),
            'gasPrice': int(5e9),
            'nonce': nonce
        })

        self.egas = web3.eth.estimate_gas(arbTxn)
        self.arbTxn = arbTxn

        pass
