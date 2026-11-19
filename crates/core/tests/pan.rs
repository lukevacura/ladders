#![cfg(not(any(loom, shuttle)))]

//! The pan matrix profile visits the window sizes in STUMPY's breadth-first order, holds for each
//! visited size exactly the two-sided profile at that size on bits at every thread count, and
//! transforms the rows as STUMPY's `pan` does: equal to STUMPY's own image on a fixed case and to
//! a plain per-cell restatement of the transform on random ones. Edge ranges are errors or
//! clamps, never panics.

use core::num::NonZeroUsize;

use ladde::batch::two_sided_profile;
use ladde::pan::{PanError, PanProfile, PanTransform, bfs_indices, max_window_size};
use ladde::quantize::Scale;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn finished(
    series: &[f64],
    min_m: usize,
    max_m: Option<usize>,
    step: usize,
    scale: Scale,
    threads: usize,
) -> PanProfile {
    let mut pan = PanProfile::new(series, min_m, max_m, step, Some(scale), nz(threads)).unwrap();
    while pan.update() {}
    pan
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn bfs_order_is_stumpys() {
    assert_eq!(bfs_indices(0), Vec::<usize>::new());
    assert_eq!(bfs_indices(1), [0]);
    assert_eq!(bfs_indices(2), [1, 0]);
    assert_eq!(bfs_indices(3), [1, 0, 2]);
    assert_eq!(bfs_indices(9), [4, 2, 7, 1, 3, 6, 8, 0, 5]);
    assert_eq!(bfs_indices(10), [5, 2, 8, 1, 4, 7, 9, 0, 3, 6]);
    for n in 0..300 {
        let mut sorted = bfs_indices(n);
        sorted.sort_unstable();
        assert!(sorted.iter().copied().eq(0..n), "n = {n}");
    }
}

/// FNV-1a over every index as eight little-endian bytes, of STUMPY's own `_bfs_indices(n)` for
/// `n` from 0 (empty) to 299.
const BFS_HASHES: [u64; 300] = [
    0xcbf29ce484222325,
    0xa8c7f832281a39c5,
    0x392209f14dea4c24,
    0x992024ad58b0cce6,
    0xa18db741c2f87405,
    0xbf1c90c3330e5d61,
    0x4b66b9d2ba969fc4,
    0x85ea5ec851b6ac02,
    0xe36899b1cb318725,
    0x81c9d6d8a142452d,
    0xb3576b44b2fc55e4,
    0xa7d3643dac15282e,
    0x527c5bf8fa11b465,
    0x9b63950831b96469,
    0x228242d2c5e99ac4,
    0x1a6ae69e7071580a,
    0x5d9cd6d09ef8e8a5,
    0x30d52932efa9f875,
    0x6db30343c0159e04,
    0x056623c2250571b6,
    0x525ac0c386ecc405,
    0x14673c87cc42fc91,
    0xbf6691375420e744,
    0xbfa0acae8b785692,
    0xdb3d5e0ef0c563c5,
    0x54e283847bdc235d,
    0x4b9cc1ef57967d44,
    0xf2de08d0e0e1cf7e,
    0x9818cbd9ebfc13a5,
    0x7b986b24ec47c0f9,
    0x987cbac37d651f44,
    0xe349aa3e2e1f0e9a,
    0x28ba230ef3c98f25,
    0xd51268e6b5bfc605,
    0x0765f6a97f7197c4,
    0x0f0fdfe52b9cc206,
    0xa706e56f029e7a25,
    0xb6ad030474ee6ce1,
    0x06a33fa0e83cd624,
    0x56b3fbe5b67b53e2,
    0x4f80bf7d430d98e5,
    0x7a263f438f1a178d,
    0x73453aa2ffc3d204,
    0xe460f7d6b93e108e,
    0xad562249feada2c5,
    0x05b851dd60f35729,
    0x9c986ddcd552ac44,
    0x2643c30c6d596faa,
    0x226afd7f7b783ac5,
    0x2023c4716892c3f5,
    0xf98e7db411848484,
    0xd01e3d589656b816,
    0x50b07e117ec7ef45,
    0x1243211622ab7251,
    0x335e461c354774a4,
    0x418d154584b9a832,
    0xf27b8105b0c726c5,
    0x1391126299ec8a3d,
    0xb99a4c8a6928ad84,
    0x67ff9892e0a63a5e,
    0x0e4ee6f8c4b723e5,
    0x59e9e984bc512919,
    0x77e7ca43ffc4d784,
    0x6c135f6f896192fa,
    0xd62e811d1cbfcee5,
    0x90947b20331c29e5,
    0xa7f8d94386e53a84,
    0xee2ac30ac8b985a6,
    0xb82560d471ac16a5,
    0xf53a4c87d44dcc01,
    0xf279a39a488305e4,
    0x39b38077d9dbdc42,
    0xc97ff3199a7f9685,
    0x331fc948fb9d926d,
    0xfea2fbb1d650e6e4,
    0x1245a1c2ec583c2e,
    0x8be066aeadfde725,
    0xbe5c31bb68178e09,
    0x8f41b9aec6e96924,
    0xdba4a8e3d0942aca,
    0xd75a1604fafaf325,
    0x4ec4d43ed8dfc9f5,
    0x23216777f5e7d904,
    0x30a6a8689b442176,
    0x5840431c0ea69525,
    0x3698563f08e5e191,
    0x62c6323580bb4f24,
    0x430933746f215412,
    0x8ac565e01509e165,
    0x456cbe849d601fbd,
    0xb9a299ff770cd6c4,
    0x8652af6d6cf3ea3e,
    0xf22f702d3d1e9745,
    0x5995ef0719a6fed9,
    0xd004c139d2c8d544,
    0xb3560cd99ec3909a,
    0x78d7cc61466aea45,
    0x7d387fdce63855e5,
    0xfe99d0031cacf484,
    0xf7e283c46c1bd086,
    0xebdd2cd146a26bc5,
    0x51cf8789b8f3aec1,
    0xb2fcf5915ebcd524,
    0xaa0e0dc96e41dc62,
    0x203462fe748bb665,
    0x66dae4c381023bed,
    0x592f84c6232bd6a4,
    0x4797b0cb9c273f0e,
    0x5c0267700ba468c5,
    0xeef51cec6e8ed209,
    0x9376e6c67affc3e4,
    0xe28c80884632986a,
    0xa06d56559addb4c5,
    0x1c23b2359c886ff5,
    0x4c3f26dce5a45484,
    0x5d6cf2e0dd0be856,
    0xe56bdbea6d1e0c85,
    0xeb69739125e51a91,
    0x6039dd09e5970ae4,
    0x0514d91069fbd072,
    0xdc60e5b6b31a01c5,
    0x4aa77097437c4b3d,
    0x360a371eee7b4b04,
    0x17853d1af72889de,
    0x7e3bf70954552e25,
    0x80caa1621a7c5259,
    0x70aef54f0663bf44,
    0x824922aa73486a7a,
    0xf363c9eafb114ae5,
    0x292db106f12b1825,
    0xfb5c1d5ad7515204,
    0x2d8aaa4ef5dcf8a6,
    0x15b65d1fa9528965,
    0x571fcb9bb9ca3cc1,
    0x3841b40599cad424,
    0xcb32a4ce04ffce42,
    0x039acd5c1d9bb345,
    0x315e1c623835df6d,
    0x58fde54bf90390a4,
    0xa4567e39fe9513ee,
    0x8e4b08cfb5659165,
    0x379b79f144674789,
    0xefb3159fde0b10e4,
    0x6e02a1b14c7b234a,
    0xfdb2e3785c4d3b85,
    0x1824a13f1ca9d035,
    0x2e48d72aa5e7b7a4,
    0xd58b68134d608c76,
    0xba12b6bf07960c05,
    0x469b44e45ddc5b51,
    0x775b08f6775731c4,
    0xda06d173a53f6212,
    0x19170630cc20f845,
    0x327ba2f46582663d,
    0x5642771b19996524,
    0xc5974f7ba8d5e67e,
    0x56a54435e23031a5,
    0x22d823f095aafc59,
    0x957ebfe9161fabe4,
    0x9687718f72ae3ada,
    0x79d3b73acb1e5825,
    0x81a71bbba3bd4e05,
    0xb0578e674c5b2ec4,
    0x3df3d84f1dfb9006,
    0x2ad47033d03cd2a5,
    0xd4afcde1780757a1,
    0xeae8673ab9a1d0e4,
    0xf0c9f7de3608e522,
    0x7c3b81f955b97805,
    0xc76163d350b9a60d,
    0x2db68a1f5e272124,
    0xa6e9a16bb013bece,
    0x41980ddfd7b65025,
    0xb54fc05c77d03069,
    0x4f62ce00cdde72e4,
    0x9e21432e4747162a,
    0x35ecad91ae1ad9e5,
    0x2e45c9fe93c05995,
    0xd17d4a7048c63204,
    0x12107007cf682ad6,
    0x98529d8f8e4f7125,
    0x08c3b2bff1c16df1,
    0x250a626426635a64,
    0xdae0ee29c9d50f72,
    0x672ba622c81c1ce5,
    0xfa3d928ae159015d,
    0x8b138ee6c16518c4,
    0x67aa79dccb55be1e,
    0xffe66bcaff4ea6c5,
    0xc56f40113f437b79,
    0xf136ef45697fda44,
    0xc192e0291a5c693a,
    0x021461a9dbefe4c5,
    0xb8b5411189608a05,
    0x7b15df2eb5005644,
    0xbf320603249283a6,
    0x2eef4ab23510af85,
    0xb81b4145a8fc7f21,
    0x1aa8a14ebe1c9224,
    0xa8c21f280eac10c2,
    0x1073d5ff92f08125,
    0x8c280fe9564b750d,
    0xc1d3c4ac66c9bb24,
    0x2ebd3944c4720a6e,
    0x376a431d10087f05,
    0x23102fd46adfb669,
    0xa327bbd1bfdd32e4,
    0xcb0dcc318eba1e8a,
    0x0beab4d5fc458165,
    0x144fcc0ac0dd5fd5,
    0xbd61124582df3824,
    0xc0c49c7e91ac95b6,
    0x94986ed9ecdcb8e5,
    0xe4de343f784758f1,
    0x63d9eb5b6ee85804,
    0xd7135271af4218d2,
    0x11fe0479b14b39a5,
    0x1aba9887c5cf6c1d,
    0xd04419e278685964,
    0x52d3d9705ac06ebe,
    0x93275b0746c74985,
    0xeedf25491db58cb9,
    0xc20c2ddfb4bab524,
    0x2b735cf2c35e961a,
    0x995cdbd7add31585,
    0x895c345c97abf3a5,
    0x3da44a465a0a8644,
    0x98f935704f177786,
    0xa2e6da6e65d32f45,
    0x9a928cd5aa4862c1,
    0x3165f729f58ce664,
    0x0ad86aa77ba640a2,
    0xe3110130518a1d65,
    0xea6e34e0e132e3ed,
    0xd9ba709d1bcc5564,
    0x1cf40cc6490967ce,
    0x3600485e38725f85,
    0x95e7f4ff63f38009,
    0xfe38dae4aec79ae4,
    0x6458fd1e48026c6a,
    0x84d8bad438f8f445,
    0xb0403b657f21bb75,
    0xc1c3ff4c6e355f44,
    0x6b1063a24f1bc596,
    0xe2ef6c977ce8b1c5,
    0x10dd8c9e399932d1,
    0x3babde7c2f0d1e64,
    0x13dd8b29c50b4b72,
    0x7649eb7381ffc705,
    0xeeeb78421d0915fd,
    0xe11156bbafa86d84,
    0x51447dabcd0865de,
    0x33774b1a0ace6d65,
    0xf275e1b19fad8659,
    0x759f8f36457ffac4,
    0x30ffea60d3fa057a,
    0xe8c4072f46af8ba5,
    0xd03ff21fe906a73e,
    0x2a5a8b3cf3de1f84,
    0xef67bb0b6b455ee5,
    0x675edc9e3eee6dbd,
    0x38f87b5ad9c4f946,
    0x245b96d85fafc35c,
    0x4b342ecf5172fef1,
    0x83f8606171f40409,
    0xbe5567f9a31924ce,
    0x7816d9e9d37950c4,
    0x59aab3faab84c5f1,
    0x4bb7ff5287093929,
    0x448d38c89936f282,
    0xf44748808006e5b0,
    0xe63a5b395994af2d,
    0xc913b7e077b654b1,
    0x41bcdc8190083f6a,
    0xd7373210c5a41d88,
    0x593b54f76f366849,
    0x99671430e95bb551,
    0x579186241fec6d96,
    0x367ea2f79f14d170,
    0xf215abab8ea93f8d,
    0x9c53ab56b9771f79,
    0x53f2c80be1ee2a72,
    0x5cb3d642172f077c,
    0x3f0d70f464db59e9,
    0xb6281f794bda49d1,
    0xe55fb8c5ac661c1a,
    0xc52d7a332823aec8,
    0x4f9c21586332709d,
    0x7744d11f8c45447d,
    0x744d4182a1811086,
    0xb80c76fbccfecea8,
    0x076e1180946c4c15,
    0x6cc537b19e0bb939,
    0x0473fb1d9b9f08be,
    0xe8aae8514cb1218c,
    0x55a59c549fe7e6c5,
    0x6a93cce5b88a7fd9,
    0x6ef255246c2305be,
    0x0d816d02046f23c8,
    0x704fb61aa926ba11,
];

fn fnv(order: &[usize]) -> u64 {
    order
        .iter()
        .flat_map(|&i| (i as u64).to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

#[test]
fn bfs_order_equals_stumpys_for_every_length_below_300() {
    for (n, &want) in BFS_HASHES.iter().enumerate() {
        assert_eq!(fnv(&bfs_indices(n)), want, "n = {n}");
    }
}

#[test]
fn largest_window_size_is_stumpys() {
    assert_eq!(max_window_size(0), 0);
    for (len, largest) in [(6, 4), (10, 7), (11, 8), (16, 12)] {
        assert_eq!(max_window_size(len), largest, "len {len}");
    }
    assert_eq!(max_window_size(100), 79);
    assert_eq!(max_window_size(300), 239);
}

const GOLDEN_SERIES: [f64; 16] = [
    0.0, 0.296875, -0.28125, -0.890625, -0.453125, -0.984375, 0.0625, 1.34375, -0.5, -0.625,
    0.484375, 0.359375, 0.109375, -0.9375, -0.03125, 0.6875,
];

/// STUMPY's `pan(binary=False)` after three updates over sizes 4 to 7 (rows 4 to 7, columns 0
/// to 15): the visited sizes are 6, 5 and 7.
const GOLDEN_SMOOTH: [f64; 64] = [
    0.11920292202211755,
    0.8617363783971559,
    0.9769029148549048,
    0.7670073295549202,
    0.2039638012047533,
    0.4787362373298253,
    0.1715803634395459,
    0.8343839176665627,
    0.24068344954767718,
    0.5318716211394505,
    0.14341217284114724,
    0.8028582752375708,
    0.997728648653434,
    0.9981631642067218,
    0.9985146795258891,
    0.9987990062871208,
    0.11920292202211755,
    0.8617363783971559,
    0.9769029148549048,
    0.7670073295549202,
    0.2039638012047533,
    0.4787362373298253,
    0.1715803634395459,
    0.8343839176665627,
    0.24068344954767718,
    0.5318716211394505,
    0.14341217284114724,
    0.8028582752375708,
    0.997728648653434,
    0.9981631642067218,
    0.9985146795258891,
    0.9987990062871208,
    0.9358872083651737,
    0.9475300000124663,
    0.8851929947964726,
    0.9847860668796082,
    0.2816745904875445,
    0.5842938860205096,
    0.9218739610773815,
    0.9571552701698905,
    0.3266437836537117,
    0.6348757234627472,
    0.9051083674381379,
    0.9934471226612918,
    0.9946963515893064,
    0.9957084585028184,
    0.9965280969715071,
    0.997191634792419,
    0.9650799213994935,
    0.9812465645527352,
    0.9919060428721775,
    0.9876659246678032,
    0.3750439055342763,
    0.6826457602366206,
    0.9715823304953524,
    0.9900061839305762,
    0.42607768945412666,
    0.7268548384151838,
    0.99902895886368,
    0.9992149173516488,
    0.9993652866272988,
    0.9994868700067759,
    0.9995851729848655,
    0.9996646498695336,
];

/// The same state through `pan(normalize=False, contrast=False, binary=False, clip=False)`.
const GOLDEN_RAW: [f64; 64] = [
    0.9408216118857009,
    1.971391845083731,
    2.4112255661139987,
    1.8656654433930384,
    1.6867472006685094,
    1.7630626898849908,
    1.5888082154978624,
    1.9610855334255033,
    1.6867472006685094,
    1.7630626898849908,
    0.9408216118857009,
    1.8656654433930384,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    0.9408216118857009,
    1.971391845083731,
    2.4112255661139987,
    1.8656654433930384,
    1.6867472006685094,
    1.7630626898849908,
    1.5888082154978624,
    1.9610855334255033,
    1.6867472006685094,
    1.7630626898849908,
    0.9408216118857009,
    1.8656654433930384,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    2.3706396340541107,
    2.3775774963827097,
    2.1831634833625686,
    2.997876865984643,
    1.9052657964693458,
    1.9720222456638379,
    2.3119320099900906,
    2.3775774963827097,
    1.9052657964693458,
    1.9720222456638379,
    2.1831634833625686,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    2.6923842135280487,
    2.880903947847185,
    3.4413187802885825,
    3.293242996756022,
    2.078290607197688,
    2.155619129604223,
    2.6923842135280487,
    3.3428144815242624,
    2.078290607197688,
    2.155619129604223,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
    3.4413187802885825,
];

fn golden_state() -> PanProfile {
    let mut pan = PanProfile::new(
        &GOLDEN_SERIES,
        4,
        Some(7),
        1,
        Some(scale(1.0 / 64.0)),
        nz(1),
    )
    .unwrap();
    assert_eq!(pan.window_sizes().collect::<Vec<_>>(), [6, 5, 7, 4]);
    for _ in 0..3 {
        assert!(pan.update());
    }
    pan
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (cell, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-12, "cell {cell}: {g} vs {w}");
    }
}

#[test]
fn the_image_equals_stumpys_on_a_fixed_case() {
    let pan = golden_state();
    let smooth = PanTransform {
        binary: false,
        ..PanTransform::default()
    };
    assert_close(&pan.pan(smooth), &GOLDEN_SMOOTH);
    let raw = PanTransform {
        normalize: false,
        contrast: false,
        binary: false,
        clip: false,
        ..PanTransform::default()
    };
    assert_close(&pan.pan(raw), &GOLDEN_RAW);
}

fn tie_heavy_series() -> Vec<(&'static str, Vec<f64>)> {
    let levels = |n: usize, k: u64| -> Vec<f64> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % k) as f64
            })
            .collect()
    };
    let periodic = |n: usize, p: usize| -> Vec<f64> {
        (0..n).map(|i| ((i % p) * (i % p) % 7) as f64).collect()
    };
    let mut runs = vec![0.0; 60];
    runs.extend(vec![3.0; 50]);
    runs.extend(levels(70, 3));
    runs.extend(vec![3.0; 40]);
    let mut gaps = levels(160, 5);
    for at in [0, 1, 2, 70, 71, 72, 73, 157, 158, 159] {
        gaps[at] = f64::NAN;
    }
    let mut repeats = levels(40, 50);
    let tail = repeats.clone();
    repeats.extend(tail.iter().map(|x| x + 100.0));
    repeats.extend(tail);
    vec![
        ("constant", vec![2.0; 150]),
        ("runs", runs),
        ("period 2", periodic(150, 2)),
        ("period 4", periodic(150, 4)),
        ("period 7", periodic(150, 7)),
        ("few levels", levels(160, 3)),
        ("repeats", repeats),
        ("gaps", gaps),
    ]
}

#[test]
fn every_row_is_the_two_sided_profile_at_its_size_whatever_the_threads() {
    for (name, series) in tie_heavy_series() {
        let base = finished(&series, 3, None, 5, scale(1.0), 1);
        assert_eq!(base.processed(), base.sizes());
        for threads in [2, 3, 8] {
            let other = finished(&series, 3, None, 5, scale(1.0), threads);
            for (a, b) in base.rows().iter().zip(other.rows()) {
                assert_eq!(bits(&a.distances), bits(&b.distances), "{name}");
                assert_eq!(a.indices, b.indices, "{name}");
            }
        }
        for row in base.rows() {
            let want = two_sided_profile(&series, nz(row.window), scale(1.0), nz(1));
            assert_eq!(row.distances.len(), want.len(), "{name} m={}", row.window);
            for (i, (_, sides)) in want.iter().enumerate() {
                let nearest = sides.nearest;
                assert_eq!(
                    row.distances[i].to_bits(),
                    nearest.squared_distance().sqrt().to_bits(),
                    "{name} m={} window {i}",
                    row.window
                );
                assert_eq!(row.indices[i], nearest.index(), "{name} m={}", row.window);
            }
        }
    }
}

#[test]
fn threaded_rows_on_a_series_long_enough_to_split_equal_the_single_thread() {
    let series: Vec<f64> = (0..3200).map(|i| ((i * i + i / 7) % 5) as f64).collect();
    let single = finished(&series, 8, Some(14), 3, scale(1.0), 1);
    let threaded = finished(&series, 8, Some(14), 3, scale(1.0), 4);
    for (a, b) in single.rows().iter().zip(threaded.rows()) {
        assert_eq!(bits(&a.distances), bits(&b.distances));
        assert_eq!(a.indices, b.indices);
    }
}

#[test]
fn rows_arrive_in_the_breadth_first_order_one_per_update() {
    let series: Vec<f64> = (0..120).map(|i| ((i * i) % 11) as f64).collect();
    let mut pan = PanProfile::new(&series, 4, Some(40), 3, Some(scale(1.0)), nz(2)).unwrap();
    let order: Vec<usize> = pan.window_sizes().collect();
    let sizes: Vec<usize> = (4..=40).step_by(3).collect();
    assert_eq!(order.len(), sizes.len());
    assert_eq!(
        order,
        bfs_indices(sizes.len())
            .into_iter()
            .map(|i| sizes[i])
            .collect::<Vec<_>>()
    );
    for (done, &window) in order.iter().enumerate() {
        assert_eq!(pan.processed(), done);
        assert!(pan.update());
        assert_eq!(pan.rows()[done].window, window);
    }
    assert!(!pan.update());
    assert_eq!(pan.processed(), order.len());
}

#[test]
fn an_inferred_scale_is_per_size_and_recorded() {
    let series: Vec<f64> = (0..90).map(|i| ((i * 7) % 13) as f64 * 0.25).collect();
    let mut pan = PanProfile::new(&series, 5, Some(9), 2, None, nz(1)).unwrap();
    while pan.update() {}
    for row in pan.rows() {
        let want = scale_of(&series, row.window);
        assert_eq!(row.scale.map(Scale::get), Some(want));
    }
}

fn scale_of(series: &[f64], window: usize) -> f64 {
    Scale::infer(series, nz(window)).unwrap().get()
}

#[test]
fn bad_ranges_are_errors_never_clamps_or_panics() {
    let series: Vec<f64> = (0..50).map(|i| ((i * 5) % 9) as f64).collect();
    let largest = max_window_size(series.len());
    let sizes = |min_m, max_m, step| {
        PanProfile::new(&series, min_m, max_m, step, Some(scale(1.0)), nz(1))
            .map(|pan| pan.window_sizes().collect::<Vec<_>>())
    };
    assert_eq!(sizes(3, None, 0), Err(PanError::ZeroStep));
    assert_eq!(sizes(3, Some(5), 1), Ok(vec![4, 3, 5]));
    assert_eq!(sizes(2, Some(5), 1), Err(PanError::WindowBelowThree));
    assert_eq!(sizes(0, Some(5), 1), Err(PanError::WindowBelowThree));
    assert_eq!(sizes(6, Some(5), 1), Err(PanError::MinAboveMax));
    assert_eq!(sizes(5, Some(5), 1), Ok(vec![5]));
    assert_eq!(sizes(3, Some(largest), 1).map(|v| v.len()), Ok(largest - 2));
    assert_eq!(
        sizes(3, Some(largest + 1), 1),
        Err(PanError::MaxAboveLargest { largest })
    );
    assert_eq!(
        sizes(3, Some(usize::MAX), usize::MAX),
        Err(PanError::MaxAboveLargest { largest })
    );
    assert_eq!(sizes(3, Some(largest), usize::MAX), Ok(vec![3]));
    let sorted = |min_m, max_m, step| {
        sizes(min_m, max_m, step).map(|mut list| {
            list.sort_unstable();
            list
        })
    };
    assert_eq!(sorted(largest + 1, None, 1), Ok(vec![40, 41]));
    assert_eq!(sorted(49, None, 1), Ok(vec![49, 50]));
    assert_eq!(sorted(3, None, 1).map(|v| v.len()), Ok(largest - 2));
    assert_eq!(sizes(50, None, 1), Err(PanError::WindowBeyondSeries));
    assert_eq!(
        sizes(usize::MAX, None, 1),
        Err(PanError::WindowBeyondSeries)
    );
    assert_eq!(sizes(0, None, 1), Err(PanError::WindowBelowThree));
    assert_eq!(sizes(2, None, 5), Err(PanError::WindowBelowThree));
    assert_eq!(
        sizes(largest + 1, Some(usize::MAX), 1),
        Err(PanError::MaxAboveLargest { largest })
    );
    assert_eq!(sizes(largest, Some(largest), 7), Ok(vec![largest]));
    let sixty = vec![1.0; 60];
    let from_47 = PanProfile::new(&sixty, 47, None, 1, Some(scale(1.0)), nz(1)).unwrap();
    assert_eq!(from_47.sizes(), 2);
    for len in 0..12 {
        let tiny = vec![1.0; len];
        for min_m in [0, 3, 4] {
            if let Ok(pan) = PanProfile::new(&tiny, min_m, None, 1, Some(scale(1.0)), nz(1)) {
                assert!(pan.window_sizes().all(|m| (3..=len).contains(&m)));
            }
        }
    }
    let nan = vec![f64::NAN; 30];
    let mut pan = PanProfile::new(&nan, 3, Some(8), 1, None, nz(1)).unwrap();
    while pan.update() {}
    assert!(pan.rows().iter().all(|r| r.scale.is_none()));
    assert!(
        pan.rows()
            .iter()
            .all(|r| r.indices.iter().all(|i| i.is_none()))
    );
    let raw = PanTransform {
        normalize: false,
        contrast: false,
        binary: false,
        clip: false,
        ..PanTransform::default()
    };
    assert!(pan.pan(raw).iter().all(|x| x.is_nan()));
}

#[test]
fn nothing_computed_is_all_nan_and_a_prefix_fills_the_rows_above() {
    let mut pan = PanProfile::new(
        &GOLDEN_SERIES,
        4,
        Some(7),
        1,
        Some(scale(1.0 / 64.0)),
        nz(1),
    )
    .unwrap();
    let raw = PanTransform {
        normalize: false,
        contrast: false,
        binary: false,
        clip: false,
        ..PanTransform::default()
    };
    assert!(pan.pan(raw).iter().all(|x| x.is_nan()));
    assert!(pan.update());
    let image = pan.pan(raw);
    let row6 = &image[2 * 16..3 * 16];
    assert_eq!(bits(&image[..16]), bits(row6));
    assert_eq!(bits(&image[16..32]), bits(row6));
    assert_eq!(bits(&image[3 * 16..]), bits(&[largest(&image); 16]));
}

fn largest(image: &[f64]) -> f64 {
    image.iter().copied().fold(f64::MIN, f64::max)
}

/// The transform restated per cell, with no shared structure: sizes ascending, rows computed or
/// not, ranks by counting.
fn reference_image(pan: &PanProfile, t: PanTransform) -> Vec<f64> {
    let width = pan.len();
    let mut by_size: Vec<(usize, usize)> = pan
        .rows()
        .iter()
        .enumerate()
        .map(|(visit, row)| (row.window, visit))
        .collect();
    by_size.sort();
    let sizes: Vec<usize> = {
        let mut all: Vec<usize> = pan.window_sizes().collect();
        all.sort();
        all
    };
    let cell = |row: usize, col: usize| -> f64 {
        let Some(&(window, visit)) = by_size.iter().find(|(w, _)| *w == sizes[row]) else {
            return f64::NAN;
        };
        let d = pan.rows()[visit].distances.get(col).copied();
        match d {
            Some(d) if d != f64::INFINITY => {
                if t.normalize {
                    let v = d * (1.0 / (2.0 * (window as f64).sqrt()));
                    if v > 1.0 { 1.0 } else { v }
                } else {
                    d
                }
            }
            _ => f64::NAN,
        }
    };
    let visit_rows: Vec<usize> = pan
        .window_sizes()
        .take(pan.processed())
        .map(|w| sizes.iter().position(|&s| s == w).unwrap())
        .collect();
    let mut image = vec![f64::NAN; sizes.len() * width];
    for &row in &visit_rows {
        for col in 0..width {
            image[row * width + col] = cell(row, col);
        }
    }
    if t.contrast {
        let flat: Vec<(usize, usize)> = visit_rows
            .iter()
            .flat_map(|&r| (0..width).map(move |c| (r, c)))
            .collect();
        let values: Vec<f64> = flat.iter().map(|&(r, c)| image[r * width + c]).collect();
        let count = flat.len();
        for (at, &(r, c)) in flat.iter().enumerate() {
            let v = values[at];
            let rank = (0..count)
                .filter(|&o| {
                    let w = values[o];
                    if v.is_nan() {
                        w.is_nan() && o < at
                    } else {
                        !w.is_nan() && (w < v || (w == v && o < at))
                    }
                })
                .count()
                + if v.is_nan() {
                    values.iter().filter(|w| !w.is_nan()).count()
                } else {
                    0
                };
            let share = if count == 1 {
                0.0
            } else if rank == count - 1 {
                1.0
            } else {
                rank as f64 * (1.0 / (count as f64 - 1.0))
            };
            image[r * width + c] = 1.0 / (1.0 + libm::exp(-10.0 * (share - t.threshold)));
        }
    }
    for &row in &visit_rows {
        for col in 0..width {
            let v = &mut image[row * width + col];
            if t.binary {
                *v = if *v <= t.threshold { 0.0 } else { 1.0 };
            }
            if t.clip && !v.is_nan() {
                *v = v.clamp(0.0, 1.0);
            }
        }
    }
    let mut computed: Vec<usize> = visit_rows.clone();
    computed.sort();
    if let Some(&last) = computed.last() {
        for row in 0..last {
            if !computed.contains(&row) {
                let source = *computed.iter().find(|&&c| c > row).unwrap();
                let copy: Vec<f64> = image[source * width..(source + 1) * width].to_vec();
                image[row * width..(row + 1) * width].copy_from_slice(&copy);
            }
        }
    }
    let top = image
        .iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a });
    for v in &mut image {
        if v.is_nan() {
            *v = top;
        }
    }
    image
}

fn transforms() -> impl Strategy<Value = PanTransform> {
    (
        prop_oneof![Just(0.2), Just(0.5), Just(0.0), Just(1.0)],
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(threshold, normalize, contrast, binary, clip)| PanTransform {
                threshold,
                normalize,
                contrast,
                binary,
                clip,
            },
        )
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    #[test]
    fn the_image_is_the_per_cell_transform_of_the_rows(
        series in prop::collection::vec(prop_oneof![Just(f64::NAN), (0u8..5).prop_map(f64::from)], 20..50),
        processed in 0usize..6,
        transform in transforms(),
    ) {
        let mut pan = PanProfile::new(&series, 3, Some(8), 1, Some(scale(1.0)), nz(1)).unwrap();
        for _ in 0..processed {
            pan.update();
        }
        let got = pan.pan(transform);
        let want = reference_image(&pan, transform);
        prop_assert_eq!(got.len(), want.len());
        for (cell, (g, w)) in got.iter().zip(&want).enumerate() {
            let close = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12;
            prop_assert!(close, "cell {}: {} vs {} ({:?})", cell, g, w, transform);
        }
    }
}
