pragma solidity ^0.6.6;
pragma experimental ABIEncoderV2;



import './interfaces/IPancakePair.sol';
import './interfaces/IERC20.sol';
import './interfaces/SafeMath.sol';
import './interfaces/SafeERC20.sol';


interface IBSCswapCallee {
    function BSCswapCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}

interface IUniswapV2Callee {
    function uniswapV2Call(address sender, uint amount0, uint amount1, bytes calldata data) external;
}
interface IThugswapCallee {
    function ThugswapCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}
interface ICheeseSwapCallee {
    function cheeseswapCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}
interface IWaultSwapCallee {
    function waultSwapCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}
interface IswapV2Callee {
    function swapV2Call(address sender, uint amount0, uint amount1, bytes calldata data) external;
}

interface IWardenCallee {
    function wardenCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}
interface IBiswapCallee {
    function BiswapCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
}

 interface IPancakeCallee {
     function pancakeCall(address sender, uint amount0, uint amount1, bytes calldata data) external;
 }

interface ChiToken {
    function freeFromUpTo(address from, uint256 value) external;
}
contract duoGasOpt {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;
    

    bool pcUnlocked = false;
    ChiToken constant private chi = ChiToken(0x0000000000004946c0e9F43F4Dee607b0eF1fA1c);

    address payable private owner;

    modifier onlyOwner() {
       require(msg.sender == owner, "cino");
        _;
    }
    
    modifier arb() {
        require (pcUnlocked, 'arb not initiated');
        
    }
    modifier discountCHI {
      uint256 gasStart = gasleft();

      _;
      uint256 initialGas = 21000 + 16 * msg.data.length;
      uint256 gasSpent = initialGas + gasStart - gasleft();
      uint256 freeUpValue = (gasSpent + 14154) / 41947;

      chi.freeFromUpTo(address(this), freeUpValue);
    }
    constructor() public {
        owner = payable(msg.sender);

    }
    function _getAmtIn(uint256 _fee, uint256 out, uint256 resIn, uint256 resOut) internal returns (uint _amtIn){
        uint256 _feeAdj = 10000 - uint256(_fee);
        uint256 num = resIn * out * 10000;
        uint256 den = (resOut - out) * _feeAdj;
        _amtIn = (num / den) + 1;

    }

    function _getAmtOut(uint256 _fee, uint256 _in, uint256 resIn, uint256 resOut) internal returns (uint _amtOut){
        uint256 _feeAdj = 10000 - uint256(_fee);
        uint256 _amtIWF = _in * _feeAdj;
        uint256 num = _amtIWF * resOut;
        uint256 den = (resIn * 10000) + _amtIWF;
        _amtOut = (num / den) - 1;

    }
    receive() external payable {}
    function _rkc(IPancakePair _pool, uint256 _swapOut0, uint256 _swapOut1, uint256 _feeAmt) internal returns(uint256 _amtIn){
        _pool.sync();
        (uint256 res0, uint256 res1,) = _pool.getReserves();
        uint256 _cp = res0 * res1;
        uint256 _balance0 = SafeMath.sub(res0, _swapOut0);
        uint256 _balance1 = SafeMath.sub(res1, _swapOut1);

        uint256 _amtIn = _getAmtIn(_feeAmt, (_swapOut0 == 0? _swapOut1:_swapOut0), (_swapOut0 == 0? res0: res1), (_swapOut0 == 0? res1:res0));
        if(_swapOut0 == 0) {
            _balance0 = _balance1 * 10000 + (_amtIn  * 10000) - (_amtIn * _feeAmt);
            _balance1 = SafeMath.mul(_balance1, 10000);
        } else if(_balance1 == res1){
            _balance0 = SafeMath.mul(_balance0, 10000);
            _balance1 = _balance1 * 10000 + (_amtIn * 10000) - (_amtIn * _feeAmt);
        } else {
            revert('f');
        }

        require(SafeMath.mul(_balance0, _balance1) >= SafeMath.mul(_cp, (10000 ** 2)), 'fk');
        return _amtIn;
    }

    function _duoSwap(uint8 _fee0, uint8 _fee1, uint8 _p1_out0, uint8 _p1_out1, uint128 _p0_out0, uint128 _p0_out1,
         address[] calldata _adds) external onlyOwner discountCHI {


        IPancakePair _pool = IPancakePair(_adds[0]);


        uint256 _amtIn = _rkc(IPancakePair(_adds[0]), _p0_out0, _p0_out1, _fee0);


        bytes memory data = abi.encode(_fee1, _p1_out0, _p1_out1, _amtIn, _adds);
        pcUnlocked = true;
        _pool.swap(_p0_out0, _p0_out1, address(this), data);
        pcUnlocked = false;

    }
     function _pancakeCall(address _sender, uint256 _amount0, uint256 _amount1, bytes memory data) internal arb {
        (uint8 _fee1, uint128 _p1_out0, uint128 _p1_out1, uint256 _amtIn0, address[] memory _adds) = abi.decode(data, (
            uint8, uint128, uint128, uint256, address[]));
       
        bytes memory _data;
        IERC20 _tokTransfer = IERC20(_adds[2]);
        IPancakePair _pool = IPancakePair(_adds[1]);

        uint256 _amtIn1 = _rkc(IPancakePair(_adds[1]), (_p1_out0 == 0? 0:_amtIn0), (_p1_out1 == 0? 0:_amtIn0), _fee1);
        require((_amount0 == 0? _amount1: _amount0) > _amtIn1, 'f2');
        require(_tokTransfer.transfer(_adds[1], _amtIn1), 'ftf');
        _pool.swap((_p1_out0 == 0? 0:_amtIn0), (_p1_out1==0?0:_amtIn0), _adds[0], _data);


    }
    function pancakeCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
       _pancakeCall(_sender, _amount0, _amount1, data);
    }

    function wardenCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function BiswapCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function swapV2Call(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function waultSwapCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
     function cheeseSwapCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function ThugswapCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function uniswapV2Call(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
    function BSCswapCall(address _sender, uint256 _amount0, uint256 _amount1, bytes calldata data) external {
        _pancakeCall(_sender, _amount0, _amount1, data);
    }
     function transferTok(uint256 _amount, address _token) external onlyOwner {
        _transferTok(_amount, _token);
    }

    function _transferTok(uint256 _amount, address _token) internal {
        IERC20 _tok = IERC20(_token);
        _tok.transfer(owner, _amount);
    }

    function transferEth(uint256 _value) external onlyOwner {
        _transferEth(_value);
    }

    function _transferEth(uint256 _value) internal {
        owner.transfer(_value);

    }

    function approve(address token, address spender, uint256 limit) external onlyOwner payable {
        _approve(token, spender, limit);
    }

    function _approve(address _token, address _spender, uint256 _limit) internal {
        IERC20 _tok = IERC20(_token);
        _tok.approve(_spender, _limit);
    }

    function massApprove(
        address[] calldata tokens,
        address[] calldata spender,
        uint256[] calldata values) external onlyOwner payable{
        for(uint256 i = 0; i < tokens.length; i++){
            IERC20 tok = IERC20(tokens[i]);
            tok.approve(spender[i], values[i]);
        }
    }

}

