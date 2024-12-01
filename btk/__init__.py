"""Personal toolkit"""

__all__ = [
    'oddf', 'evenf', 'normalize', 'standardize', 'criticals',
    'smooth_avg', 'bin_to_num', 'stretch_fit', 'half_point', 'permutes',
    'around_point', 'area', 'fzip', 'lrsort', 'rrsort', 'cprint', 'hprint'
]

from typing import Iterable, Container
from numbers import Number
from numpy import float32, array, mean, std, gradient, min as nmin, max as nmax, round as nround

COLORS = [*[f'\x1b[{y};{x};{z}m' for z in (22, 1) for x in (49, *(z for z in range(41, 46))) for y in (*(z for z in range(30, 37)), 97) if not (y in (30, 97) and x == 49) and x - 10 != y], '\x1b[30;47;1m']
RST = '\x1b[0m'
SNUMS = list('0123456789')


def oddf(num: int) -> int:
    """Forces a number to be odd"""
    if num % 2 == 0:
        num += 1
    return int(num)

def evenf(num: int) -> int:
    """Forces a number to be even"""
    if num % 2 != 0:
        num += 1
    return int(num)

def normalize(data: Iterable[Number]) -> list:
    """
    Normalize input values

    (x - x.min) / (x.max - x.min)

    Args:
        data (Iterable): Input Data

    Returns:
        [type]: Normalized input data
    """
    return array((data - nmin(data)) / (nmax(data) - nmin(data)), dtype='float32')

def standardize(data: Iterable[Number]) -> list:
    """
    Standardize input values

    (x - x.mean) / x.standard_deviation

    Args:
        data (Iterable): Input data

    Returns:
        [type]: Standardized input data
    """
    return array((data - mean(data)) / std(data), dtype=float32)

def smooth_avg(data: Iterable[Number]) -> list:
    """
    Generate a smoothed version of a data set where each point is replaced by the average of itself and immeadiately adjacent points

    Args:
        data (list): A list of continuous data points

    Returns:
        list: A smoother list of continuous data points
    """
    smoothed = []
    for i, x in enumerate(data):
        if i == 0:
            smoothed.append((x + data[i + 1]) / 2)
        elif i == len(data) - 1:
            smoothed.append((x + data[i - 1]) / 2)
        else:
            smoothed.append((data[i - 1] + x + data[i + 1]) / 3)
    return smoothed

def criticals(data: Iterable[Number], idx: bool = False) -> list:
    """
    Create a list of critical points of a continuous data set
    Critical Points: Maxima, Minima, Gradient Maxima, Gradient Minima, Gradient Roots

    Args:
        data (list): A list of continuous data points
        idx (bool, optional): A custom index. Defaults to False.

    Returns:
        list: A list of tuples that contains the index of a critical point and the critical point type
    """
    grads = gradient(data)
    grads2 = gradient(grads)
    crits = []
    if not idx:
        idx = range(len(data))
    """ else:
        idx = [round((x[1] - x[0]) / 2) + x[0] for x in idx] """
    for i, x in enumerate(idx, 1):
        if i > len(idx) - 2:
            break
        if data[i - 1] < data[i] and data[i + 1] < data[i]:
            crits.append((x, 'max'))
        if data[i - 1] > data[i] and data[i + 1] > data[i]:
            crits.append((x, 'min'))
        if grads[i] > 0 and grads[i + 1] < 0 or grads[i] < 0 and grads[i + 1] > 0:
            crits.append((x, 'dzero'))
        if grads[i - 1] < grads[i] and grads[i + 1] < grads[i]:
            crits.append((x, 'dmax'))
        if grads[i - 1] > grads[i] and grads[i + 1] > grads[i]:
            crits.append((x, 'dmin'))
        if grads2[i] > 0 and grads2[i + 1] < 0 or grads2[i] < 0 and grads2[i + 1] > 0:
            crits.append((x, 'ddzero'))
        if grads2[i - 1] < grads2[i] and grads2[i + 1] < grads2[i]:
            crits.append((x, 'ddmax'))
        if grads2[i - 1] > grads2[i] and grads2[i + 1] > grads2[i]:
            crits.append((x, 'ddmin'))
    return crits

def half_point(num1: Number, num2: Number):
    """
    Gives the halfway point between input numbers

    Args:
        num1 (intorfloat): A number
        num2 (intorfloat): A number

    Returns:
        [type]: Halfway point number
    """
    if num2 > num1:
        mid = ((num2 - num1) / 2) + num1
    else:
        mid = ((num1 - num2) / 2) + num2
    return mid

def around_point(num: Number, step: Number) -> tuple[int, int]:
    """
    Gives the points around a number seperated by the step size

    Args:
        num ([type]): number
        step ([type]): distance from input number

    Returns:
        [type]: tuple containing the points surrounding the input
    """
    return (num - step, num + step)

def area(coords: Iterable[Number]):
    """Find area of a box when given the box corners in a tuple"""
    return (coords[1] - coords[0]) * (coords[3] - coords[2])

def bin_to_num(inp: Iterable[int]) -> int:
    """
    Convert a binary number into integer

    Args:
        inp (list): Binary number

    Returns:
        int: Integer representation of binary input
    """
    if isinstance(inp, str):
        inp = [int(x) for x in inp]
    if isinstance(inp, (int, float)):
        inp = [int(x) for x in str(round(inp))]
    if set(inp) != set(1, 0):
        return None
    bin_state = 0
    for x in inp[::-1]:
        if x == 1:
            total += 2**bin_state
            bin_state += 1
        else:
            bin_state += 1
    return total

def stretch_fit(inp: Iterable[Number], top: int, dtype: type ='uint8'):
    """
    Stretch data to fit from 0 to arg(top)

    Args:
        inp (Iterable[Number]): Input data array
        top (int): Maximum value of the output data
        dtype (type, optional): Datatype of the output. Defaults to 'uint8'.

    Returns:
        [type]: Stretched data array
    """
    mnm = nmin(inp)
    inp = inp - mnm
    mxm = nmax(inp)
    inp = nround(inp * (top / mxm)).astype(dtype)
    return inp

def permutes(items, lb=1, ub=0):
    if not ub: ub = len(items)
    out = [[x] for x in items]
    for _ in range(lb, ub): out.extend([[*x, y] for y in items for x in out])
    return set([tuple(sorted(set(x), reverse=True)) for x in out])

def fzip(*cols: Iterable[Iterable], pad: str='bottom') -> list[tuple]:
    """
    zip() except inputs of unequal length will be filled with ''

    Args:
        pad (str): Selection for padding the top or bottom. "bottom" or "top". Defaults to "bottom".

    Returns:
        list[tuple]: List of input iterables but padded with '' to equal length.
    """
    if isinstance(cols[0], (int, str, float)): return [cols]
    top = 0
    for x in cols:
        if len(x) > top: top = len(x)
    out = []
    for i in range(top):
        pack = []
        for x in cols:
            if pad == 'bottom':
                if i < top - len(x): pack.append('')
                else: pack.append(x[i-(top-len(x))])
            else:
                if i >= len(x): pack.append('')
                else: pack.append(x[i])
        out.append(tuple(pack))
    return out

def hprint(text: str, hltxt: str | Container='', c: str='\x1b[32m', r_str:bool=False) -> None | str:
    """
    Highlight text in a string

    Args:
        text (str): Text corpus
        hltxt (str | Container, optional): Highlighting target text.
        c (str, optional): Color Selection. Defaults to '\x1b[32m'.
        r_str (bool, optional): Option to return text instead of printing. Defaults to False.
    """
    if not hltxt:
        if r_str: return f'{c}{text}{RST}'
        else: print(f'{c}{text}{RST}')
    elif isinstance(hltxt, str):
        ils, idx, pl = [], 0, len(hltxt)
        while hltxt in text[idx:]:
            idx = text.index(hltxt, idx)
            ils.append((idx, idx+pl))
            idx += pl
        if ils:
            ils.append((len(text),))
            out = text[0:ils[0][0]]
            for i, x in enumerate(ils[:-1]):
                out += f'{c}{text[x[0]:x[1]]}{RST}{text[x[1]:ils[i+1][0]]}'
            if r_str: return out
            else: print(out)
        else:
            if r_str: return text
            else: print(text)
    elif isinstance(hltxt, tuple) and len(hltxt) == 2 and isinstance(hltxt[0], int) and isinstance(hltxt[1], int) and hltxt[0] < hltxt[1]:
        text = str(text)
        ils = get_ndex(text, get_nranged(hltxt[0], hltxt[1]))
        if ils:
            ils.append((len(text),))
            out = text[0:ils[0][0]]
            for i, x in enumerate(ils[:-1]):
                out += f'{c}{text[x[0]:x[1]]}{RST}{text[x[1]:ils[i+1][0]]}'
            if r_str: return out
            else: print(out)
        else:
            if r_str: return text
            else: print(text)
    elif isinstance(hltxt, Container):
        if len(hltxt) > 82: raise ValueError("Maximum highlight hltxts exceeded")
        hltxt = [get_nranged(p[0], p[1]) if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int) and isinstance(p[1], int) and p[0] < p[1] else str(p) for i, p in enumerate(hltxt)]
        ilsx = [0 for _ in enumerate(text)]
        for pid, p in enumerate(hltxt):
            if isinstance(p, str):
                idx, pl = 0, len(p)
                while p in text[idx:]:
                    idx = text.index(p, idx)
                    for i in range(idx, idx+pl):
                        if ilsx[i] == 0: ilsx[i] = pid+1
                        elif ilsx[i] > 0: ilsx[i] = -1
                    idx += pl
            elif isinstance(p, dict):
                ndx = get_ndex(text, p)
                for i in ndx:
                    for j in range(i[0], i[1]):
                        if ilsx[j] == 0: ilsx[j] = pid+1
                        elif ilsx[j] > 0: ilsx[j] = -1
        cc, ls, ils = 0, 0, []
        for i, x in enumerate(ilsx):
            if cc != x:
                ils.append((ls, i, cc))
                cc, ls = x, i
        ils.append((ls, len(ilsx), cc))
        out = ''
        for x in ils:
            if x[2] == 0: out += text[x[0]:x[1]]
            else: out += f'{COLORS[x[2]]}{text[x[0]:x[1]]}{RST}'
        if r_str: return out
        else: print(out)

def cprint(
    data: Container[Iterable], 
    pos: Container[int] or int = 0, 
    halign: str or dict[int: str] = 'l', 
    valign: str='top', 
    col_width: int=16, 
    newline: bool=False, 
    spc_share: bool=True,
    trim: str='', 
    hlite: str | dict | Container='',
    hl_clr: int=2,
    hl_col: bool=False
    ) -> None:
    """
    Print data in columns with highlighting capabilities.

    Args:
        data (Container[Iterable]): Data to be printed. Must be an Container of Iterables or a Container of Containers containing iterables.
        pos (Container[int]orint, optional): Iterable defining the position to print a column at in tabs. Defaults to 2 tabs from the previous column.
        halign (_type_, optional): Alignment for columns of data. 'l', 'm', or 'r' for left middle and right aligned. Individual column alignment can be defined with a dictionary. Defaults to 'l'.
        valign (str, optional): Selection for aligning against the top or bottom. "bottom" or "top". Defaults to "top".
        col_width (int, optional): Number of character spaces to designate as a column. Defaults to 16.
        newline (bool, optional): Print a newline after each row. Defaults to False.
        spc_share (bool, optional): Allow columns facing in the same direction to share space. Defaults to True.
        trim (str, optional): Trim entries exceeding column width. False will not trim, 'l' trims from the left, 'r' trims from the right. Defaults to False.
        hlite (str | dict | Container, optional): Highlight text globally or by column. Entries correspond to a single column by default. Defaults to None.
        hl_clr (int, optional): Highlight color specified with an integer value between 0 and 8. Defaults to 2: green.
        hl_col (bool, optional): Highlight strings by column rather than for every line. Defaults to False.

    """
    if not isinstance(pos, (int, Container)) or (isinstance(pos, Container) and any(not isinstance(y, int) for y in pos)): raise ValueError(f'Invalid Argument: pos {pos}')
    if not isinstance(col_width, int): raise ValueError(f'Invalid Argument: col_width {col_width}')
    if isinstance(pos, int): pos = [(col_width if halign == 'r' and pos == 0 else pos)]
    if not isinstance(data, str) and len(pos) > len(data): pos = pos[:len(data)]
    if hlite and isinstance(hlite, str): hlite = [hlite]

    if hlite:
        if not isinstance(hl_clr, int) or hl_clr not in (0, 1, 2, 3, 4, 5, 6, 7): raise ValueError(f'Invalid color selection {hl_clr}')
        hl_clr = f'\x1b[3{hl_clr}m'

        if not hl_col:
            if not isinstance(hlite, (tuple, list, set, bool)): raise ValueError('Invalid hlite argument for per line highlighting')
            if hlite is not True: hlite = [x if isinstance(x, tuple) and len(x) == 2 and  isinstance(x[0], int) and  isinstance(x[1], int) and  x[0] < x[1] else str(x) for x in hlite]
        else:
            if isinstance(hlite, dict) and not all(isinstance(y, int) for y in hlite): raise ValueError('Invalid dictionary keys for hlite. Keys must be integers corresponding to columns')
            if isinstance(hlite, (tuple, list)): hdct = {i: {'s': x if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], int) and  isinstance(x[1], int) and  x[0] < x[1] else str(x)} for i, x in enumerate(hlite)}
            else: hdct = {i: {'s': hlite} for i, _ in enumerate(data)}
            if isinstance(hlite, str):
                for i in hdct: hdct[i]['c'] = hl_clr
            else:
                for i in hdct: hdct[i]['c'] = COLORS[i]

    col_dict = {}
    if halign in ('l', 'm', 'r'): last_align = halign
    else: last_align = 'l'
    for i, x in enumerate(pos[:len(data)+1]):
        col_dict[i] = {'p': x * col_width}
        if isinstance(halign, dict) and i in halign and halign[i] in ('l', 'm', 'r'): col_dict[i]['a'] = last_align = halign[i]
        else: col_dict[i]['a'] = last_align
    if col_dict[0]['a'] == 'r': col_dict[0]['g'] = last_gap = col_dict[0]['p']
    elif 1 in col_dict: col_dict[0]['g'] = last_gap = (col_dict[1]['p'] - col_dict[0]['p']) // 2 if (col_dict[0]['a'] == 'm' or not spc_share) else col_dict[1]['p'] - col_dict[0]['p']
    else: col_dict[0]['g'] = last_gap = col_width
    if col_dict[0]['g'] == 0: raise ValueError('Column at position 0 can not be right aligned')
    col_dict[0]['sp'] = col_dict[0]['p'] if col_dict[0]['a'] in ('l', 'm') else (col_dict[0]['p'] - col_dict[0]['g'])
    col_dict[0]['ep'] = (col_dict[0]['p'] + col_dict[0]['g']) if col_dict[0]['a'] in ('l', 'm') else col_dict[0]['p']


    if not isinstance(data, Container) or isinstance(data, str):
        if hlite is True: hprint(f"{col_dict[0]['p'] * ' '}{data}", c=hl_clr)
        elif hlite: hprint(f"{col_dict[0]['p'] * ' '}{data}", hlite)
        else: print(f"{col_dict[0]['p'] * ' '}{data}")
    elif (isinstance(data, Container) and not isinstance(data, str)) and len(data) == 1 and not (isinstance(data[0], Container) and not isinstance(data[0], str)):
        if hlite is True:
            for x in data: hprint(f"{col_dict[0]['p'] * ' '}{x}", c=hl_clr)
        elif hlite:
            for x in data: hprint(f"{col_dict[0]['p'] * ' '}{x}", hlite)
        else:
            for x in data: print(f'{col_dict[0]["p"] * " "}{x}')

    else:
        for i, _ in enumerate(data):
            if i == 0: continue
            if i not in col_dict:
                col_dict[i] = {'a': (halign if halign in ('l', 'm', 'r') else (halign[i] if isinstance(halign, dict) and i in halign else last_align))}
                last_align = col_dict[i]['a']
            if col_dict[i]['a'] in ('l', 'm'):
                if 'p' not in col_dict[i]:
                    col_dict[i]['p'] = (col_dict[i-1]['p'] + col_dict[i-1]['g'] if col_dict[i-1]['a'] in ('l', 'm') else col_dict[i-1]['p'])
                if i+1 in col_dict:
                    if (not spc_share or col_dict[i]['a'] == 'm') and col_dict[i+1]['a'] == 'r':
                        col_dict[i]['g'] = last_gap = (col_dict[i+1]['p'] - col_dict[i]['p']) // 2
                    else: col_dict[i]['g'] = last_gap = col_dict[i+1]['p'] - col_dict[i]['p']
                else: col_dict[i]['g'] = last_gap
                col_dict[i]['sp'] = col_dict[i]['p']
                col_dict[i]['ep'] = (col_dict[i]['p'] + col_dict[i]['g'])
            else:
                if 'p' not in col_dict[i]:
                    if col_dict[i-1]['a'] in ('l', 'm'):
                        col_dict[i]['p'] = col_dict[i-1]['p'] + col_dict[i-1]['g'] + last_gap
                    else: col_dict[i]['p'] = col_dict[i-1]['p'] + last_gap
                if col_dict[i-1]['a'] == 'r' or (spc_share and col_dict[i-1]['a'] == 'l'):
                    col_dict[i]['g'] = last_gap = col_dict[i]['p'] - col_dict[i-1]['p']
                else: col_dict[i]['g'] = last_gap = col_dict[i]['p'] - col_dict[i-1]['p'] - col_dict[i-1]['g']
                col_dict[i]['sp'] = (col_dict[i]['p'] - col_dict[i]['g'])
                col_dict[i]['ep'] = col_dict[i]['p']
            if col_dict[i]['a'] == 'r' and col_dict[i-1]['a'] == 'l' and spc_share: col_dict[i-1]['share'] = True
            if i == len(data)-1 and col_dict[i]['a'] == 'l': col_dict[i]['g'] = 1024
            if col_dict[i]['g'] < 1: raise ValueError(f'Invalid positions argument {pos}')

        data = fzip(*data, pad=valign)
        for x in data:
            pod = {}
            pod[0] = (col_dict[0]['p'] * ' ' if col_dict[0]['a'] in ('l', 'm') else '')
            is_skip = False
            for j, y in enumerate(x):
                if not is_skip:
                    sc, al, space, pid = str(y), col_dict[j]['a'], col_dict[j]['g'] - 1, 0
                    if 'share' in col_dict[j]:
                        is_skip, lsc, rsc, dspc = True, sc, str(x[j+1]), space // 2
                        while lsc or rsc:
                            if len(lsc) + len(rsc) < space:
                                pod[pid] += f'{lsc}{" " * (space-(len(lsc)+len(rsc)))}{rsc}'
                                lsc, rsc = '', ''
                            else:
                                if not lsc:
                                    if trim == 'r': pod[pid] += f' {rsc[-space:]}'
                                    else: pod[pid] += f' {rsc[:space]}'
                                    rsc = rsc[space:]
                                elif not rsc:
                                    pod[pid] += lsc[:space]
                                    lsc = lsc[space:]
                                else:
                                    if len(lsc) < dspc:
                                        if trim == 'r': pod[pid] += f'{lsc} {rsc[-(space - (len(lsc)+1)):]}'
                                        else: pod[pid] += f'{lsc} {rsc[:(space - (len(lsc)+1))]}'
                                        lsc, rsc = '', rsc[(space - (len(lsc)+1)):]
                                    elif len(rsc) < dspc: 
                                        pod[pid] += f'{lsc[:(space - (len(rsc)+1))]} {rsc}'
                                        lsc, rsc = lsc[(space - (len(rsc)+1)):], ''
                                    else:
                                        if trim == 'r': pod[pid] += f'{lsc[-dspc:]} {rsc[-dspc:]}'
                                        else: pod[pid] += f'{lsc[:dspc]} {rsc[:dspc]}'
                                        lsc, rsc = lsc[dspc:], rsc[dspc:]
                                if trim: lsc, rsc = '', ''
                                else: pid += 1
                                if pid not in pod: pod[pid] = col_dict[j]['p'] * ' '
                    else:
                        while sc:
                            if len(sc) <= space:
                                if al == 'r' or pid > 0: pod[pid] += f'{(space-len(sc)) * " "}{sc}'
                                elif al == 'l': pod[pid] += (sc if j == len(x)-1 else f'{sc}{(space-len(sc)) * " "}')
                                else: pod[pid] += f'{((space-len(sc)) // 2) * " "}{sc}{(space-len(sc)-((space-len(sc)) // 2)) * " "}'
                            else:
                                if trim == 'r': pod[pid] += sc[-space:]
                                else: pod[pid] += sc[:space]
                            if trim: sc = ''
                            else: sc = sc[space:]
                            if sc:
                                pid += 1
                                if pid not in pod: pod[pid] = (col_dict[j]['p'] * ' ' if al in ('l', 'm') else (col_dict[j]['p']-col_dict[j]['g']) * ' ')   
                else: is_skip = False

                if j < len(x)-1:
                    if col_dict[j]['a'] in ('l', 'm'): end = (col_dict[j]['p']+col_dict[j]['g'])
                    elif col_dict[j+1]['a'] in ('l', 'm'): end = col_dict[j+1]['p']
                    else: end = col_dict[j]['p']
                    for z in pod: pod[z] += (end - len(pod[z])) * " "

            if not hlite:
                for lrow in pod.values(): print(lrow.rstrip())
            elif hlite is True:
                for lrow in pod.values(): hprint(lrow.rstrip(), c=hl_clr)
            else:
                if not hl_col:
                    for lrow in pod.values(): hprint(lrow.rstrip(), hlite)
                else:
                    for lrow in pod.values():
                        skip2 = False
                        if col_dict[0]['sp'] != 0: cstr = col_dict[0]['sp'] * " "
                        else: cstr = ''
                        for ck, cv in col_dict.items():
                            if skip2:
                                skip2 = False
                                continue
                            if ck in hdct:
                                if 'share' in cv:
                                    skip2 = True
                                    cpair = lrow[cv['sp']:cv['ep']]
                                    ridx = cpair[:-1].rindex(' ')
                                    cstr += hprint(cpair[:ridx], hdct[ck]['s'], hdct[ck]['c'], r_str=True)
                                    cstr += hprint(cpair[ridx:], hdct[ck+1]['s'], hdct[ck+1]['c'], r_str=True) + " "
                                else: cstr += hprint(lrow[cv['sp']:cv['ep']], hdct[ck]['s'], hdct[ck]['c'], r_str=True)
                            else: cstr += lrow[cv['sp']:cv['ep']]
                        print(cstr.rstrip())
            if newline: print('\n')

def get_ndex(txt, pdct):
    its = []
    txt = str(txt)
    tln = len(txt)
    for i, x in enumerate(txt):
        if x not in pdct: continue
        if -1 in pdct[x]:
            its.append((i, i+1))
            continue
        tc, j = pdct[x], 0

        while i+j < tln-1:
            j += 1
            if 'w' in tc:
                k = tc['w']
                while k and i+j < tln and txt[i+j] in SNUMS:
                    k -= 1
                    j += 1
                its.append((i, i+j))
                break
            elif txt[i+j] in tc:
                if tc[txt[i+j]] is True:
                    its.append((i, i+j+1))
                    break
                else: tc = tc[txt[i+j]]
            else: break

    if len(its) > 1:
        its = its[::-1]
        merge = []
        old = its.pop()
        while its:
            new = its.pop()
            if old[1] >= new[0]:
                old = (old[0], new[1])
            else:
                merge.append(old)
                old = new
        merge.append(old)
        its = merge
    return its

def get_nranged(lb: int, ub: int):
    if lb >= ub: raise ValueError('Invalid bounds')
    pdct = {}
    while lb <= ub:
        slb = str(lb)
        if lb % 10 != 0 or lb == 0:
            if lb >= 10:
                t = pdct
                while slb and slb[0] in t:
                    t = t[slb[0]]
                    slb = slb[1:]
                if len(slb) > 1: t[slb[0]] = {slb[-1]: {}}
                elif isinstance(t, list): t.append(slb[0])
                else: t[slb[0]] = {}
            else: pdct[slb] = {-1: True}
            lb += 1
        else:
            nzero = len(slb) - len(slb.rstrip('0'))
            if lb + (10**nzero) > ub:
                while lb + (10**nzero) > ub and nzero > 0:
                    nzero -= 1
            if nzero > 0:
                k = str(int(lb * (10**-nzero)))
                t = pdct
                while k:
                    if k[0] in t: t = t[k[0]]
                    else: t[k[0]] = t = {}
                    k = k[1:]
                t['w'] = nzero
                lb += 10**nzero
            else:
                t = pdct
                while slb and slb[0] in t:
                    t = t[slb[0]]
                    slb = slb[1:]
                if len(slb) > 1: t[slb[0]] = {slb[-1]: {}}
                elif isinstance(t, list): t.append(slb[0])
                else: t[slb[0]] = {}
                lb += 1
    return traverse(pdct, {}, True)

def traverse(dct, f='', r=''):
    for x in dct.items():
        if isinstance(x[1], dict):
            if len(x[1]) > 0:
                dct[x[0]] = traverse(x[1], f, r)
            elif len(x[1]) == 0:
                dct[x[0]] = True
    return dct

def rdx_sort(inp: Iterable, reverse: bool=False, mcd: bool=False) -> list[str]:
    """
    Radix sort for strings

    Args:
        inp (list[str]): List of strings to be sorted
        reverse (bool, optional): Order ascending. Defaults to False.
        mcd (bool, optional): Most Common Denominator / Right most characters are the most significant. Defaults to False.
    
    Returns:
        list[str]: Sorted list of strings
    """

    def rdx_cnt(inp, out, pos, ivals, counts):
        inpk = []
        for x in inp:
            try: inpk.append(x[-pos])
            except IndexError: 
                inpk.append(x[0])
        for x in inpk:
            counts[x] += 1
        for i, x in enumerate(ivals[1:]):
            counts[x] += counts[ivals[i]]
        for i in range(len(inp)-1, -1, -1):
            counts[inpk[i]] -= 1
            out[counts[inpk[i]]] = inp[i]
        return out

    out = [0 for _ in inp]
    ivals = set()
    for x in inp: ivals = ivals | set(x)
    ivals = sorted(ivals, reverse=reverse)
    if ivals[0] == "'": ivals.append(ivals.pop(0))
    counts = {x: 0 for x in ivals}
    end_point = max({len(x) for x in inp})
    pos = 1

    if mcd:
        end_point -= 1
        pos -= 1
    while pos <= end_point:
        if mcd:
            inp = rdx_cnt(inp, out.copy(), -pos, ivals, counts.copy())
        else:
            inp = rdx_cnt(inp, out.copy(), pos, ivals, counts.copy())
        pos += 1
    return inp

def rrsort(data: Container, trim: bool=True):
    if not data: return []
    cap = max([len(x) for x in data])
    if trim: return [x.strip() for x in rdx_sort([f'{(cap-len(x)) * " "}{x}' for x in data], mcd=True)]
    else: return rdx_sort([f'{(cap-len(x)) * " "}{x}' for x in data], mcd=True)

def lrsort(data: Container, trim: bool=True):
    if not data: return []
    cap = max([len(x) for x in data])
    if trim: return [x.strip() for x in rdx_sort([f'{x}{(cap-len(x)) * " "}' for x in data])]
    else: return rdx_sort([f'{x}{(cap-len(x)) * " "}' for x in data])

def remove_from(inp1, inp2):
    """Removes all items in arg1 from arg2"""
    if isinstance(inp2, list):
        return [x for x in inp2 if x not in inp1]
    if isinstance(inp2, set):
        return inp2 - set(inp1)
    if isinstance(inp2, dict):
        return {x[0]: x[1] for x in inp2.items() if x[0] not in x[1]}
    if isinstance(inp2, tuple):
        return (x for x in inp2 if x not in inp1)
    else:
        return False

def trim_outliers(inlst: Iterable[Number], var_window: float=0.5, min_incnts: int=8, min_outcnts: int=3) -> list[float]:
    """
    Trim outliers from an input list by expanding an acceptance window from the middle of the input list

    Args:
        inlst (list[float]): Input iterator
        var_window (float, optional): Acceptance window while iterating. Percentage of average and 1/2 of standard deviation. Defaults to 0.5.
        min_incnts (int, optional): minimum inputs. Defaults to 8.
        min_outcnts (int, optional): minimum outputs. Defaults to 3.

    Returns:
        (list[float]): Input list but with outliers trimmed
    """
    if len(inlst) < min_incnts:
        return False
    inlst = sorted(inlst)
    outlst = [inlst.pop(round(len(inlst)/2) - 1)]
    outlst.append(inlst.pop(round(len(inlst)/2) - 1))
    while inlst:
        item = inlst.pop(round(len(inlst)/2) - 1)
        if abs(item - mean(outlst)) <= max(4 * var_window * std(outlst), var_window * mean(outlst)):
            outlst.append(item)
    if len(outlst) >= min_outcnts:
        return outlst

class Record:
    """
    Decorator that records all inputs and outputs for a functions calls

    Records can be saved to different labels by adding a keyword argument "record=label_name" when calling the function.
    This label name will be used to reference record for recall and deletion.
    """

    def __init__(self, fn):
        self.fn = fn
        self.history = {'default': {}}

    def __call__(self, *args, **kwargs):
        if 'record' in kwargs:
            label = kwargs.pop('record')
            try:
                record = self.history[label]
            except KeyError:
                self.history[label] = {}
                record = self.history[label]
        else:
            record = self.history['default']
        out = self.fn(*args, **kwargs)
        try:
            record[max(record)+1] = ((*args, *tuple([f'{x[0]}: {x[1]}' for x in kwargs.items()])), out)
        except ValueError:
            record[0] = ((*args, *tuple([f'{x[0]}: {x[1]}' for x in kwargs.items()])), out)
        return out

    def recall(self, last=0, label=False):
        """
        Prints recorded data

        Args:
            last (int, optional): Number of records to print. Prints all records if no argument is provided.
            label (bool, optional): Label records to print from.
        """
        if label:
            record = self.history[label]
        else:
            record = self.history['default']
        end = max(record)
        if not last:
            for x in record:
                print(f'input:\t{record[x][0]}\noutput:\t{record[x][1]}\n')
        else:
            for x in range(max(0, end-last), end, 1):
                print(f'input:\t{record[x][0]}\noutput:\t{record[x][1]}\n')

    def clear(self, last=0, label=False):
        """
        Prints recorded data

        Args:
            last (int, optional): Number of records to delete. Deletes all records if no argument is provided.
            label (bool, optional): Label records to delete from. Deletes from default record if no argument is provided.
        """
        if not last:
            if not label:
                self.history = {'default': {}}
            else:
                self.history.pop(label)
            return
        if label:
            record = self.history[label]
        else:
            record = self.history['default']
        end = max(record)+1
        for x in range(max(0, (end-(last))), end, 1):
            record.pop(x)

class Incr:
    """Custom generator that can be iterated by calling it"""
    def __init__(self, start: int = 0, gap: int = 1):
        self.val = start
        self.gap = gap
        self.gen = self.inc_func()

    def inc_func(self):
        while True:
            yield self.val
            self.val += self.gap

    def __iter__(self):
        return self.gen

    def __call__(self):
        return next(self.gen)

    def __repr__(self):
        return str(self.val)
