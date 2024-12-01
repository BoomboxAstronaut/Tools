#![allow(unused)]
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;


#[pymodule]
fn lvsearch(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lsearch, m)?)?;
    Ok(())
}

fn pyconvy(obj: &PyAny) -> PyResult<Vec<String>> {
	let rustobj: PyResult<Vec<String>> = obj.iter()?.map(|i| i.and_then(PyAny::extract::<String>)).collect();
	rustobj
}


#[pyfunction]
fn lsearch(root: String, #[pyo3(from_py_with = "pyconvy")] mut corpus: Vec<String>, distance: Option<usize>, edgeguard: Option<bool>, addend: Option<bool>) -> Result<Vec<String>, PyErr> {

    let rlen: usize = root.len();
    if rlen < 3 { return Err(PyValueError::new_err("Words shorter than 2 character lengths are invalid")) }
    if distance.unwrap_or(3) > 7 { return Err(PyValueError::new_err("Edit distances greater than 7 are invalid")) }
    let mut wlmax: usize = root.len() + distance.unwrap_or(3);
    let wlmin: usize;
	if addend.unwrap_or(false) { wlmax += 2; }
    if rlen - distance.unwrap_or(3) < 4 { wlmin = 3 }
    else { wlmin = rlen - distance.unwrap_or(3) }


    let mut edged: HashMap<usize, Vec<usize>> = HashMap::new();
    if edgeguard.unwrap_or(true) {
        for i in 2..50 {
            if i < 4 { edged.insert(i, vec![i-1]); }
            else if i == 4 { edged.insert(i, vec![i-2, i-1]); }
            else if i == 5 { edged.insert(i, vec![i-3, i-2, i-1]); }
            else if i > 5 && i < 8 { edged.insert(i, vec![i-4, i-3, i-2, i-1]); }
            else if i > 7 && i < 12 { edged.insert(i, vec![i-5, i-4, i-3, i-2, i-1]); }
            else if i > 11 && i < 16 { edged.insert(i, vec![i-6, i-5, i-4, i-3, i-2, i-1]); }
            else { edged.insert(i, vec![i-7, i-6, i-5, i-4, i-3, i-2, i-1]); }
        }
    }

    let mut kmin: usize = (root.len() / 2);
    if rlen < 4 {kmin += 1}
    let rchars: Vec<char> = Vec::from_iter(root.chars());
    corpus.retain(|x|
        x.len() >= wlmin
        && x.len() <= wlmax
        && &rchars.iter().filter(|&y| x.contains(*y)).count() >= &kmin
    );


    let mut ckeys: Vec<Vec<char>> = Vec::new();
    for x in permutes(vec!['r', 'a', 't', 'd'], 1, distance.unwrap_or(3)) {
        let mut newx: Vec<char> = Vec::new();
        let mut rcnt: usize = 0;
        let mut acnt: usize = 0;
        let mut tcnt: usize = 0;
        let mut dcnt: usize = 0;
        for y in x {
            if y == 'r' { rcnt += 1; }
            else if y == 'a' { acnt += 1; }
            else if y == 't' {
                if tcnt > 0 && newx[newx.len()-1] == 't' { continue }
                else { tcnt += 1; }
            } else { dcnt += 1; }
            if (rlen+acnt) - dcnt < kmin { break }
            else { newx.push(y); }
        }
        if !ckeys.contains(&newx) { ckeys.push(newx); }
    }


    let mut stage0: HashSet<String> = HashSet::new();
    let mut call_cache: HashMap<(char, usize, usize, String), Vec<(String, usize)>> = HashMap::new();
    let mptv: Vec<usize> = Vec::new();
    for mut code in ckeys {
        let mut stage1: Vec<(String, usize)> = vec![(root.clone(), 0)];
        while code.len() > 0 {
            let key: char = code.pop().unwrap();
            let mut dp: usize = 1;
            while code.len() > 0 && code[code.len()-1] == key {
                code.pop();
                dp += 1;
            }

            let mut stage2: Vec<(String, usize)> = Vec::new();
            for (word, rpos) in stage1 {
				if word.len() < 3 && key == 'd' { continue }
                let mapkey: (char, usize, usize, String) = (key.clone(), dp.clone(), rpos.clone(), word.clone());
                let mut ekey: &Vec<usize>;
                if edgeguard.unwrap_or(true) { ekey = &edged[&word.len()]; }
                else { ekey = &mptv; }

                let mut result: Vec<(String, usize)>;
                if call_cache.contains_key(&mapkey) {
                    result = call_cache[&mapkey].clone();
                } else {
                    if key == 'r' {
                        result = replacement(word.to_string(), dp, rpos, ekey);
                    } else if key == 'a' {
                        result = insertion(word.to_string(), dp, rpos, ekey);
                    } else if key == 't' {
                        result = transposition(word.to_string(), rpos, ekey);
                    } else {
                        result = deletion(word.to_string(), dp, rpos, ekey);
                    }
                    call_cache.insert(mapkey, result.clone());
                }
                result.retain(|x| !stage2.contains(&x));
                stage2.extend(result);
            }
            stage2.retain(|x| x.0.len() - x.0.matches('.').collect::<Vec<&str>>().len() >= kmin);
            stage1 = stage2;
        }
        stage0.extend(match_wildcards(stage1, &corpus, addend.unwrap_or(false)).into_iter());
    }
    return Ok(Vec::from_iter(stage0.drain()))
}


fn insertion(word: String, distance: usize, root_pos: usize, edgeguard: &Vec<usize>) -> Vec<(String, usize)> {
    let mut output: Vec<(String, usize)> = Vec::new();
    let mut widx: Vec<usize> = Vec::from_iter(0..word.len()+1);
    if edgeguard.len() > 0 { widx.retain(|x| x == &0 || edgeguard.contains(&(x-1))); }
    let skeys: Vec<Vec<usize>> = permutes(widx, 1, distance);

    for sk in skeys {
        let mut wvar: String = word.clone();
        let mut new_root: usize = root_pos;
        for idx in sk {
            wvar.insert(idx, '.');
            if idx <= new_root { new_root += 1; }
        }
        let pack: (String, usize) = (wvar, new_root);
        if !output.contains(&pack) { output.push(pack); }
    }
    return output
}


fn deletion(word: String, distance: usize, root_pos: usize, edgeguard: &Vec<usize>) -> Vec<(String, usize)> {
    let mut output: Vec<(String, usize)> = Vec::new();
    let mut widx: Vec<usize> = Vec::from_iter(0..word.len());
    if edgeguard.len() > 0 { widx.retain(|x| edgeguard.contains(x)); }
    let skeys: Vec<Vec<usize>> = unqsort(permutes(widx, 1, distance));
    for sk in skeys {
        let mut wvar = word.clone();
        let mut new_root: usize = root_pos;
        for idx in sk {
            if idx != root_pos {
                if idx == wvar.len() { wvar.pop(); }
                else { wvar.remove(idx); }
                if idx < new_root { new_root -= 1; }
            }
        }
        let pack: (String, usize) = (wvar, new_root);
        if !output.contains(&pack) { output.push(pack); }
    }
    return output
}


fn transposition(word: String, root_pos: usize, edgeguard: &Vec<usize>) -> Vec<(String, usize)> {
    let mut output: Vec<(String, usize)> = Vec::new();
    let mut widx: Vec<usize> = Vec::from_iter(0..word.len()-1);
    widx.remove(root_pos);
    if root_pos > 0 { widx.remove(&root_pos - 1); }
    if edgeguard.len() > 0 { widx.retain(|x| edgeguard.contains(&(x+1))); }

    for i in widx {
        let mut wvar = word.clone();
        let c0: char = wvar.remove(i);
        let c1: char;
        if wvar.len() == i { c1 = wvar.pop().unwrap(); }
        else { c1 = wvar.remove(i); }
        wvar.insert(i, c0);
        wvar.insert(i, c1);
        let pack: (String, usize) = (wvar, root_pos);
        if !output.contains(&pack) { output.push(pack); }
    }
    return output
}


fn replacement(word: String, distance: usize, root_pos: usize, edgeguard: &Vec<usize>) -> Vec<(String, usize)> {
    let mut output: Vec<(String, usize)> = Vec::new();
    let mut skeys: Vec<Vec<usize>>;
    if edgeguard.len() > 0 {
        skeys = unqsort(permutes(Vec::from_iter(edgeguard[0]..word.len()), 1, distance));
    } else {
        skeys = unqsort(permutes(Vec::from_iter(0..word.len()), 1, distance));
    }
    for sk in skeys {
        let mut wvar: String = String::new();
        for i in 0..word.len() {
            if sk.contains(&i) && i != root_pos {
                wvar.push_str(".");
            } else {
                wvar.push_str(word.get(i..i+1).unwrap());
            }
        }
        let pack: (String, usize) = (wvar, root_pos);
        if !output.contains(&pack) { output.push(pack); }
    }
    return output
}


fn permutes<T: Clone+Copy>(input: Vec<T>, lower: usize, upper: usize) -> Vec<Vec<T>> {
    let mut output: Vec<Vec<T>> = Vec::new();
    for x in &input {
        output.push(vec![*x]);
    }

    for _ in lower..upper {
        let mut sub_out: Vec<Vec<T>> = Vec::new();
        for m in &output {
            for o in &input {
                let mut pack: Vec<T> = Vec::from_iter(m.clone().into_iter());
                pack.push(*o);
                sub_out.push(pack);
            }
        }
        output.extend(sub_out);
    }
    return output
}

fn unqsort<T: Ord+PartialEq>(items: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let mut output: Vec<Vec<T>> = Vec::new();
    for mut x in items {
        let xl = x.len();
        x.sort();
        x.dedup();
        x.reverse();
        if x.len() == xl && !output.contains(&x) {
            output.push(x);
        }
    }
    return output
}

fn match_wildcards(mut words: Vec<(String, usize)>, wdict: &Vec<String>, addend: bool) -> Vec<String> {
    let mut output: Vec<String> = Vec::new();
    for word_pack in words {
        let mut word: String = word_pack.0;
		if addend { word.push_str(".."); }
        let widx: Vec<usize> = Vec::from_iter(word.chars().into_iter().enumerate().filter(|(_, x)| x != &'.').map(|(j, _)| j));
        let mut sub_out: Vec<String> = Vec::new();

        for dwrd in wdict {
            if dwrd.len() == word.len() && widx.iter().all(|i| dwrd[*i..i+1] == word[*i..i+1]) {
                sub_out.push(dwrd.to_string());
            }
        }
        sub_out.retain(|x| !output.contains(&x));
        output.extend(sub_out);
    }
    return output
}