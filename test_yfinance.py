import yfinance as yf
import traceback
print('yfinance version:', getattr(yf, '__version__', 'unknown'))
try:
    t = yf.Ticker('AAPL')
    hist = t.history(period='5d')
    print('history rows:', len(hist))
    try:
        print(hist.tail(1).to_dict())
    except Exception:
        print('history printed; conversion failed')
except Exception as e:
    print('ERROR:')
    traceback.print_exc()
