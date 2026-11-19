//! Prints one hash per engine output over a fixed seeded corpus, so two targets can be compared
//! line by line; route counts print as plain numbers.
//!
//! To check two targets agree, run
//! `cargo run -p testkit --example target_corpus --release --features ladde/parallel`
//! once per `--target` (or machine) and diff the outputs, ignoring lines that start with `arch`
//! or `info ` (counts that follow the machine's core count), e.g.
//! `diff <(grep -v -e '^arch' -e '^info ' a.txt) <(grep -v -e '^arch' -e '^info ' b.txt)`.
//! Every other line is the same on any core count.
#[cfg(not(any(loom, shuttle)))]
mod corpus {
    use core::num::NonZeroUsize;

    use ladde::batch::multi::{
        maamp_report, maamp_report_tiled, mstump_report, mstump_report_tiled,
    };
    use ladde::batch::{
        BatchReport, SpillReport, ab_join_single_threaded, ab_join_spill_report, discords, fluss,
        left_profile, left_profile_report_tiled, left_profile_spill_report, left_topk_profile,
        left_topk_profile_tiled, nonnorm, two_sided_profile, two_sided_profile_tiled,
        two_sided_spill_report, two_sided_topk_profile, two_sided_topk_profile_tiled,
    };
    use ladde::bounded::BoundedStream;
    use ladde::entry::{Entry, TwoSided, TwoSidedTopk};
    use ladde::floss::Floss;
    use ladde::index::SampleIndex;
    use ladde::motif::{MotifParams, motifs};
    use ladde::multidim::{
        DimensionDistance, MultiError, MultiProfile, exact_decisions, screened_cells,
    };
    use ladde::narrow::RouteCounts;
    use ladde::nonnorm::{NonNormNeighbor, NonNormTwoSided, NonNormTwoSidedTopk};
    use ladde::nonnorm_bounded::BoundedNonNormStream;
    use ladde::nonnorm_stream::GrowingNonNormStream;
    use ladde::pattern::{
        MaxDistance, PatternLibrary, PatternStream, distance_profile, find_matches,
    };
    use ladde::quantize::Scale;
    use ladde::route::{Fired, RouteForce, SpillForce, forced_route};
    use ladde::stream::GrowingStream;
    use testkit::rng::XorShift64;

    struct Hash(u64);
    impl Hash {
        fn new() -> Self {
            Hash(0xcbf29ce484222325)
        }
        fn u64(&mut self, v: u64) {
            for b in v.to_le_bytes() {
                self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        fn entry(&mut self, e: Entry) {
            self.u64(e.rho().to_bits());
            self.u64(e.squared_distance().to_bits());
            self.u64(e.index().get().unwrap_or(u64::MAX));
        }
        fn row(&mut self, row: u64, e: Entry) {
            self.u64(row);
            self.entry(e);
        }
        fn bytes(&mut self, b: &[u8]) {
            self.u64(b.len() as u64);
            for &x in b {
                self.u64(u64::from(x));
            }
        }
        fn f64s(&mut self, v: &[f64]) {
            self.u64(v.len() as u64);
            for x in v {
                self.u64(x.to_bits());
            }
        }
    }

    fn nz(x: usize) -> NonZeroUsize {
        NonZeroUsize::new(x).unwrap()
    }

    fn walk(seed: u64, n: usize, step: i64) -> Vec<i64> {
        let mut r = XorShift64(seed);
        let mut level = 0i64;
        (0..n)
            .map(|_| {
                level += r.below(2 * step as u64 + 1) as i64 - step;
                level
            })
            .collect()
    }

    fn ticks(v: Vec<i64>) -> Vec<f64> {
        v.into_iter().map(|t| t as f64 * 0.01).collect()
    }

    fn corpus() -> Vec<(&'static str, Vec<f64>, Option<Scale>)> {
        let tick = Scale::new(0.01);
        let mut out = Vec::new();
        out.push(("walk", ticks(walk(1, 700, 3)), tick));
        let mut r = XorShift64(7);
        let ties: Vec<f64> = (0..700)
            .map(|i| {
                if (i / 3 + r.below(2) as usize).is_multiple_of(2) {
                    0.0
                } else {
                    0.5
                }
            })
            .collect();
        out.push(("ties", ties, Scale::new(0.5)));
        let periodic: Vec<f64> = (0..700).map(|i| [0.0, 1.0, 2.0, 1.0][i % 4]).collect();
        out.push(("periodic", periodic, Scale::new(1.0)));
        let mut gaps = ticks(walk(3, 700, 3));
        let mut r = XorShift64(11);
        for x in gaps.iter_mut() {
            if r.below(12) == 0 {
                *x = f64::NAN;
            }
        }
        for x in gaps[300..320].iter_mut() {
            *x = f64::NAN;
        }
        out.push(("nangap", gaps, tick));
        out.push((
            "offset1e6",
            ticks(walk(5, 700, 4))
                .into_iter()
                .map(|x| x + 1.0e6)
                .collect(),
            tick,
        ));
        out.push((
            "offset1e9",
            ticks(walk(5, 700, 4))
                .into_iter()
                .map(|x| x + 1.0e9)
                .collect(),
            tick,
        ));
        let mut r = XorShift64(13);
        let plain: Vec<f64> = (0..700)
            .map(|i| {
                let v =
                    (i as f64 * 0.21).sin() * 3.0 + (r.below(2001) as i64 - 1000) as f64 * 0.0005;
                (v * 10000.0).round() / 10000.0
            })
            .collect();
        let inferred = Scale::infer(&plain, nz(12));
        out.push(("plain_double", plain, inferred));
        let mut shock = walk(17, 700, 2);
        for (i, t) in shock.iter_mut().enumerate() {
            if i == 250 || i == 500 {
                *t += 3_000_000;
            }
        }
        out.push(("shock", ticks(shock), tick));
        out.push(("walk4000", ticks(walk(23, 4000, 5)), tick));
        let mut spikes = walk(31, 700, 3);
        spikes[200] += 1_000_000_000_000;
        spikes[450] -= 700_000_000_000;
        spikes[451] += 3_000_000_000;
        out.push(("spikes", ticks(spikes), tick));
        let mut excursions = walk(37, 700, 3);
        for k in 0..9 {
            for t in &mut excursions[30 + k * 70..70 + k * 70] {
                *t += 300_000_000_000;
            }
        }
        out.push(("excursions", ticks(excursions), tick));
        out
    }

    fn p(label: &str, h: Hash) {
        println!("{label} {:016x}", h.0);
    }

    fn routes(label: &str, c: RouteCounts) {
        println!(
            "{label} narrow {} wide {} reanchors {} switches {}",
            c.narrow_updates, c.wide_updates, c.reanchors, c.switches_to_wide
        );
    }

    fn rows(h: &mut Hash, v: &[(u64, Entry)]) {
        h.u64(v.len() as u64);
        for &(r, e) in v {
            h.row(r, e);
        }
    }

    fn stream_rows(h: &mut Hash, v: &[(SampleIndex, Entry)]) {
        h.u64(v.len() as u64);
        for &(r, e) in v {
            h.row(r.get().unwrap_or(u64::MAX), e);
        }
    }

    fn fired(label: &str, f: Fired) {
        println!(
            "{label} fired wide {} reanchors {} off_center {} per_tile {} spill_on {} spill_off {} sparse_on {} sparse_off {}",
            f.wide_switches,
            f.extra_reanchors,
            f.off_center_anchors,
            f.per_tile_series,
            f.spill_forced_on,
            f.spill_forced_off,
            f.sparse_forced_on,
            f.sparse_forced_off
        );
    }

    /// A run on every core labels its counts `info`: they follow the machine's core count.
    fn info(threads: &str) -> &'static str {
        if threads == "all" { "info " } else { "" }
    }

    fn spill(label: &str, threads: &str, r: SpillReport) {
        println!(
            "{}{label} spill plan {:?} engaged {} outliers {} spilled {} cells {} chains {}",
            info(threads),
            r.plan.map(|p| (p.engage, p.natural)),
            r.engaged,
            r.outlier_ticks,
            r.spilled_windows,
            r.walked.cells,
            r.walked.chains
        );
    }

    /// Hashes a report from a fixed thread count or tiling; prints one from every core as `info`.
    fn report(h: &mut Hash, label: &str, threads: &str, r: impl Into<Option<BatchReport>>) {
        let Some(r) = r.into() else {
            return;
        };
        if threads == "all" {
            println!("info {label} tiles {} wide {}", r.tiles, r.wide_tiles);
        } else {
            h.u64(r.tiles as u64);
            h.u64(r.wide_tiles as u64);
        }
    }

    fn sides(h: &mut Hash, v: &[(u64, TwoSided)]) {
        h.u64(v.len() as u64);
        for &(r, s) in v {
            h.row(r, s.left);
            h.entry(s.right);
            h.entry(s.nearest);
        }
    }

    fn topk_sides(h: &mut Hash, v: &[(u64, TwoSidedTopk)]) {
        h.u64(v.len() as u64);
        for (r, s) in v {
            h.row(*r, s.left);
            h.entry(s.right);
            h.u64(s.nearest.len() as u64);
            for &e in &s.nearest {
                h.entry(e);
            }
        }
    }

    /// Runs `run` once per thread count under `force`, printing what the force changed.
    fn per_threads(
        label: &str,
        force: RouteForce,
        all: usize,
        mut run: impl FnMut(&'static str, NonZeroUsize),
    ) {
        for (threads, t) in [("t1", 1), ("t3", 3), ("all", all)] {
            let g = forced_route(force);
            run(threads, nz(t));
            fired(&format!("{}{label} {threads}", info(threads)), g.fired());
        }
    }

    fn batch_forces() -> [(&'static str, RouteForce); 4] {
        let none = RouteForce::default();
        [
            ("natural", none),
            ("wide", RouteForce { wide: true, ..none }),
            (
                "spill_on",
                RouteForce {
                    spill: SpillForce::On,
                    ..none
                },
            ),
            (
                "spill_off",
                RouteForce {
                    spill: SpillForce::Off,
                    ..none
                },
            ),
        ]
    }

    /// The z-normalized batch profiles under every batch route; `t3` runs a fixed tiling.
    fn znorm_routes(tag: &str, series: &[f64], n: NonZeroUsize, scale: Scale, all: usize) {
        let other = ticks(walk(29, 500, 3));
        let tick = Scale::new(0.01).expect("valid scale");
        let (b, c, l) = (nz(3), nz(2), nz(5));
        for (fname, force) in batch_forces() {
            per_threads(&format!("{tag} znorm[{fname}]"), force, all, |th, t| {
                let label = format!("{tag} left[{fname}] {th}");
                let mut h = Hash::new();
                if th == "t3" {
                    let (v, rep) = left_profile_report_tiled(series, n, scale, t, b, c, l);
                    rows(&mut h, &v);
                    report(&mut h, &label, th, rep);
                } else {
                    let (v, rep, sp) = left_profile_spill_report(series, n, scale, t);
                    rows(&mut h, &v);
                    report(&mut h, &label, th, rep);
                    spill(&label, th, sp);
                }
                p(&label, h);

                let label = format!("{tag} two_sided[{fname}] {th}");
                let mut h = Hash::new();
                if th == "t3" {
                    let (v, rep) = two_sided_profile_tiled(series, n, scale, t, b, c, l);
                    sides(&mut h, &v);
                    report(&mut h, &label, th, rep);
                } else {
                    let (v, rep, sp) = two_sided_spill_report(series, n, scale, t);
                    sides(&mut h, &v);
                    report(&mut h, &label, th, rep);
                    spill(&label, th, sp);
                }
                p(&label, h);

                for k in [1, 3] {
                    let label = format!("{tag} two_sided_topk{k}[{fname}] {th}");
                    let mut h = Hash::new();
                    let run = if th == "t3" {
                        two_sided_topk_profile_tiled(series, n, scale, t, b, c, l, nz(k))
                            .map(|(v, r)| (v, Some(r)))
                    } else {
                        two_sided_topk_profile(series, n, scale, t, nz(k)).map(|v| (v, None))
                    };
                    match run {
                        Ok((v, rep)) => {
                            topk_sides(&mut h, &v);
                            report(&mut h, &label, th, rep);
                        }
                        Err(_) => h.u64(u64::MAX),
                    }
                    p(&label, h);
                }

                let label = format!("{tag} ab_join[{fname}] {th}");
                let mut h = Hash::new();
                let (v, rep, sp) = ab_join_spill_report(series, &other, n, scale, tick, t);
                rows(&mut h, &v);
                report(&mut h, &label, th, rep);
                spill(&label, th, sp);
                let (v, rep, sp) = ab_join_spill_report(&other, series, n, tick, scale, t);
                rows(&mut h, &v);
                report(&mut h, &label, th, rep);
                spill(&format!("{label} reversed"), th, sp);
                p(&label, h);
            });
        }
    }

    fn nn(h: &mut Hash, e: NonNormNeighbor, scale: Scale) {
        let sq = e.squared_ticks();
        h.u64(sq as u64);
        h.u64((sq >> 64) as u64);
        h.u64(e.index().get().unwrap_or(u64::MAX));
        h.u64(e.euclidean(scale).to_bits());
    }

    fn nn_kept(h: &mut Hash, v: &[NonNormNeighbor], scale: Scale) {
        h.u64(v.len() as u64);
        for &e in v {
            nn(h, e, scale);
        }
    }

    fn nn_rows(h: &mut Hash, v: &[(u64, NonNormNeighbor)], scale: Scale) {
        h.u64(v.len() as u64);
        for &(r, e) in v {
            h.u64(r);
            nn(h, e, scale);
        }
    }

    fn nn_sides(h: &mut Hash, v: &[(u64, NonNormTwoSided)], scale: Scale) {
        h.u64(v.len() as u64);
        for &(r, s) in v {
            h.u64(r);
            nn(h, s.left, scale);
            nn(h, s.right, scale);
            nn(h, s.nearest, scale);
        }
    }

    fn nn_topk_sides(h: &mut Hash, v: &[(u64, NonNormTwoSidedTopk)], scale: Scale) {
        h.u64(v.len() as u64);
        for (r, s) in v {
            h.u64(*r);
            nn(h, s.left, scale);
            nn(h, s.right, scale);
            nn_kept(h, &s.nearest, scale);
        }
    }

    fn nn_join_topk(h: &mut Hash, v: &[(u64, Vec<NonNormNeighbor>)], scale: Scale) {
        h.u64(v.len() as u64);
        for (r, es) in v {
            h.u64(*r);
            nn_kept(h, es, scale);
        }
    }

    fn wide_forces() -> [(&'static str, RouteForce); 2] {
        [
            ("natural", RouteForce::default()),
            (
                "wide",
                RouteForce {
                    wide: true,
                    ..RouteForce::default()
                },
            ),
        ]
    }

    /// The non-normalized batch profiles under both of their routes; `t3` runs a fixed tiling.
    fn nonnorm_batch(tag: &str, series: &[f64], n: NonZeroUsize, scale: Scale, all: usize) {
        let other = ticks(walk(29, 500, 3));
        let (b, c, l) = (nz(3), nz(2), nz(5));
        for (fname, force) in wide_forces() {
            per_threads(&format!("{tag} nonnorm[{fname}]"), force, all, |th, t| {
                let tiled = th == "t3";
                let label = format!("{tag} nonnorm_left[{fname}] {th}");
                let mut h = Hash::new();
                let (v, rep) = if tiled {
                    nonnorm::left_profile_report_tiled(series, n, scale, t, b, c, l)
                } else {
                    nonnorm::left_profile_report(series, n, scale, t)
                };
                nn_rows(&mut h, &v, scale);
                report(&mut h, &label, th, rep);
                p(&label, h);

                let label = format!("{tag} nonnorm_two_sided[{fname}] {th}");
                let mut h = Hash::new();
                if tiled {
                    let (v, rep) = nonnorm::two_sided_profile_tiled(series, n, scale, t, b, c, l);
                    nn_sides(&mut h, &v, scale);
                    report(&mut h, &label, th, rep);
                } else {
                    nn_sides(
                        &mut h,
                        &nonnorm::two_sided_profile(series, n, scale, t),
                        scale,
                    );
                }
                p(&label, h);

                for k in [1, 3] {
                    let label = format!("{tag} nonnorm_two_sided_topk{k}[{fname}] {th}");
                    let mut h = Hash::new();
                    let run = if tiled {
                        nonnorm::two_sided_topk_profile_tiled(series, n, scale, t, b, c, l, nz(k))
                            .map(|(v, r)| (v, Some(r)))
                    } else {
                        nonnorm::two_sided_topk_profile(series, n, scale, t, nz(k))
                            .map(|v| (v, None))
                    };
                    match run {
                        Ok((v, rep)) => {
                            nn_topk_sides(&mut h, &v, scale);
                            report(&mut h, &label, th, rep);
                        }
                        Err(_) => h.u64(u64::MAX),
                    }
                    p(&label, h);
                }

                let label = format!("{tag} nonnorm_ab_join[{fname}] {th}");
                let mut h = Hash::new();
                for (x, y) in [(series, other.as_slice()), (other.as_slice(), series)] {
                    let (v, rep) = if tiled {
                        nonnorm::ab_join_report_tiled(x, y, n, scale, t, b, c, l)
                    } else {
                        nonnorm::ab_join_report(x, y, n, scale, t)
                    };
                    nn_rows(&mut h, &v, scale);
                    report(&mut h, &label, th, rep);
                    let run = if tiled {
                        nonnorm::ab_join_topk_tiled(x, y, n, scale, t, b, c, l, nz(3))
                            .map(|(v, r)| (v, Some(r)))
                    } else {
                        nonnorm::ab_join_topk(x, y, n, scale, t, nz(3)).map(|v| (v, None))
                    };
                    match run {
                        Ok((v, rep)) => {
                            nn_join_topk(&mut h, &v, scale);
                            report(&mut h, &label, th, rep);
                        }
                        Err(_) => h.u64(u64::MAX),
                    }
                }
                p(&label, h);
            });
        }
    }

    fn nn_out(h: &mut Hash, v: &[(SampleIndex, NonNormNeighbor)], scale: Scale) {
        h.u64(v.len() as u64);
        for &(r, e) in v {
            h.u64(r.get().unwrap_or(u64::MAX));
            nn(h, e, scale);
        }
    }

    /// The non-normalized streams under both of their routes, one and three neighbors per row,
    /// and restored halfway from a checkpoint.
    fn nonnorm_streams(tag: &str, series: &[f64], n: NonZeroUsize, scale: Scale) {
        let w = nz(64);
        let (head, tail) = series.split_at(350);
        for (fname, force) in wide_forces() {
            let g = forced_route(force);
            let mut s = GrowingNonNormStream::new(n, scale);
            let mut topk = GrowingNonNormStream::with_k(n, scale, nz(3));
            let mut h = Hash::new();
            let mut out = Vec::new();
            s.update_many(head, &mut out);
            let mut restored = GrowingNonNormStream::restore(&s.checkpoint()).expect("restore");
            s.update_many(tail, &mut out);
            nn_out(&mut h, &out, scale);
            let mut out = Vec::new();
            restored.update_many(tail, &mut out);
            nn_out(&mut h, &out, scale);
            h.bytes(&s.checkpoint());
            h.bytes(&restored.checkpoint());
            for &x in series {
                if let Some((r, es)) = topk.update_topk(x) {
                    h.u64(r.get().unwrap_or(u64::MAX));
                    nn_kept(&mut h, es, scale);
                }
            }
            h.bytes(&topk.checkpoint());
            p(&format!("{tag} nonnorm_growing[{fname}]"), h);
            routes(
                &format!("{tag} nonnorm_growing_routes[{fname}]"),
                s.route_counts(),
            );

            let mut s = BoundedNonNormStream::new(n, w, scale).expect("w >= m");
            let mut topk = BoundedNonNormStream::with_k(n, w, scale, nz(3)).expect("w >= m");
            let mut h = Hash::new();
            let mut out = Vec::new();
            s.update_many(head, &mut out);
            let mut restored = BoundedNonNormStream::restore(&s.checkpoint()).expect("restore");
            s.update_many(tail, &mut out);
            nn_out(&mut h, &out, scale);
            let mut out = Vec::new();
            restored.update_many(tail, &mut out);
            nn_out(&mut h, &out, scale);
            h.bytes(&s.checkpoint());
            h.bytes(&restored.checkpoint());
            for &x in series {
                if let Some((r, es)) = topk.update_topk(x) {
                    h.u64(r.get().unwrap_or(u64::MAX));
                    nn_kept(&mut h, es, scale);
                }
            }
            h.bytes(&topk.checkpoint());
            p(&format!("{tag} nonnorm_bounded[{fname}]"), h);
            routes(
                &format!("{tag} nonnorm_bounded_routes[{fname}]"),
                s.route_counts(),
            );
            fired(&format!("{tag} nonnorm_streams[{fname}]"), g.fired());
        }
    }

    fn multi_hash<D: DimensionDistance>(
        h: &mut Hash,
        label: &str,
        threads: &str,
        r: Result<(MultiProfile<D>, BatchReport), MultiError>,
    ) {
        match r {
            Ok((profile, rep)) => {
                for (ds, is) in profile.distance.iter().zip(&profile.index) {
                    h.u64(ds.len() as u64);
                    for (d, i) in ds.iter().zip(is) {
                        h.u64(d.to_f64().to_bits());
                        h.u64(i.get().unwrap_or(u64::MAX));
                    }
                }
                report(h, label, threads, rep);
            }
            Err(e) => h.bytes(format!("{e:?}").as_bytes()),
        }
    }

    /// Hashes `run`, then prints the cells the screen settled and the orders the exact compare
    /// decided; on more than one thread they follow which thread takes which tile, so `info`.
    fn multi_run(label: &str, threads: &str, run: impl FnOnce(&mut Hash)) {
        let (screened, exact) = (screened_cells(), exact_decisions());
        let mut h = Hash::new();
        run(&mut h);
        p(label, h);
        let info = if threads == "t1" { "" } else { "info " };
        println!(
            "{info}{label} screened {} exact {}",
            screened_cells() - screened,
            exact_decisions() - exact
        );
    }

    /// mstump and maamp over a few dimensions of corpus series, under both of their routes;
    /// `t3` runs a fixed tiling, and `t1` also runs tiles of one row and lane.
    fn multi(corpus: &[(&'static str, Vec<f64>, Option<Scale>)], all: usize) {
        let pick = |name: &str| {
            corpus
                .iter()
                .find(|c| c.0 == name)
                .map(|c| (c.1.as_slice(), c.2.expect("corpus scale")))
                .expect("corpus series")
        };
        let tick = Scale::new(0.01).expect("valid scale");
        let inputs: [&[&str]; 3] = [
            &["walk", "ties", "nangap"],
            &["offset1e6", "shock"],
            &["spikes", "periodic", "walk"],
        ];
        let variants: [(&[usize], bool); 4] =
            [(&[], false), (&[1], false), (&[], true), (&[1, 0], true)];
        for names in inputs {
            let dims: Vec<&[f64]> = names.iter().map(|&s| pick(s).0).collect();
            let scales: Vec<Scale> = names.iter().map(|&s| pick(s).1).collect();
            for m in [5usize, 12] {
                let n = nz(m);
                for (include, discords) in variants {
                    let tag = format!(
                        "multi {}/m{m} include{include:?} discords{discords}",
                        names.join("+")
                    );
                    for (fname, force) in wide_forces() {
                        per_threads(&format!("{tag} [{fname}]"), force, all, |th, t| {
                            let label = format!("{tag} mstump[{fname}] {th}");
                            multi_run(&label, th, |h| {
                                let r = if th == "t3" {
                                    mstump_report_tiled(
                                        &dims,
                                        n,
                                        &scales,
                                        include,
                                        discords,
                                        t,
                                        nz(3),
                                        nz(2),
                                        nz(5),
                                    )
                                } else {
                                    mstump_report(&dims, n, &scales, include, discords, t)
                                };
                                multi_hash(h, &label, th, r);
                            });
                            let label = format!("{tag} maamp[{fname}] {th}");
                            multi_run(&label, th, |h| {
                                let r = if th == "t3" {
                                    maamp_report_tiled(
                                        &dims,
                                        n,
                                        tick,
                                        include,
                                        discords,
                                        t,
                                        nz(3),
                                        nz(2),
                                        nz(5),
                                    )
                                } else {
                                    maamp_report(&dims, n, tick, include, discords, t)
                                };
                                multi_hash(h, &label, th, r);
                            });
                            if th == "t1" {
                                let label = format!("{tag} mstump_tiled[{fname}] t1 1/1/1");
                                multi_run(&label, th, |h| {
                                    let r = mstump_report_tiled(
                                        &dims, n, &scales, include, discords, t, t, t, t,
                                    );
                                    multi_hash(h, &label, th, r);
                                });
                                let label = format!("{tag} maamp_tiled[{fname}] t1 1/1/1");
                                multi_run(&label, th, |h| {
                                    let r = maamp_report_tiled(
                                        &dims, n, tick, include, discords, t, t, t, t,
                                    );
                                    multi_hash(h, &label, th, r);
                                });
                            }
                        });
                    }
                }
            }
        }
    }

    pub fn main() {
        let all = std::thread::available_parallelism().map_or(4, NonZeroUsize::get);
        println!("arch {}", std::env::consts::ARCH);
        let corpus = corpus();
        for (name, series, scale) in &corpus {
            let (name, series) = (*name, series.as_slice());
            let Some(scale) = *scale else {
                println!("{name} no-scale");
                continue;
            };
            for m in [5usize, 12] {
                let n = nz(m);
                let tag = format!("{name}/m{m}");
                let mut h = Hash::new();
                rows(&mut h, &left_profile(series, n, scale));
                p(&format!("{tag} left"), h);
                for (name, threads, band, chunk, tile) in [
                    ("t1", 1, 8, 4, 4),
                    ("t3", 3, 3, 2, 5),
                    ("all", all, 64, 8, 64),
                    ("t7", 7, 1, 1, 1),
                ] {
                    let mut h = Hash::new();
                    let (v, rep) = left_profile_report_tiled(
                        series,
                        n,
                        scale,
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                    );
                    rows(&mut h, &v);
                    h.u64(rep.tiles as u64);
                    h.u64(rep.wide_tiles as u64);
                    p(&format!("{tag} left_tiled {name} {band}/{chunk}/{tile}"), h);
                }
                for (name, threads) in [("t1", 1), ("all", all)] {
                    let mut h = Hash::new();
                    sides(&mut h, &two_sided_profile(series, n, scale, nz(threads)));
                    p(&format!("{tag} two_sided {name}"), h);
                }
                let mut h = Hash::new();
                for (r, s) in
                    two_sided_profile_tiled(series, n, scale, nz(3), nz(3), nz(2), nz(5)).0
                {
                    h.row(r, s.left);
                    h.entry(s.right);
                    h.entry(s.nearest);
                }
                p(&format!("{tag} two_sided_tiled"), h);
                for k in [1usize, 3] {
                    let mut h = Hash::new();
                    for (r, es) in left_topk_profile(series, n, scale, nz(all), nz(k)) {
                        h.u64(r);
                        h.u64(es.len() as u64);
                        for e in es {
                            h.entry(e);
                        }
                    }
                    p(&format!("{tag} topk{k}"), h);
                    let mut h = Hash::new();
                    let tiled = left_topk_profile_tiled(
                        series,
                        n,
                        scale,
                        nz(3),
                        nz(3),
                        nz(2),
                        nz(5),
                        nz(k),
                    )
                    .0;
                    for (r, es) in tiled {
                        h.u64(r);
                        for e in es {
                            h.entry(e);
                        }
                    }
                    p(&format!("{tag} topk{k}_tiled"), h);
                }
                let mut h = Hash::new();
                for d in discords(series, n, scale, nz(all), 3) {
                    h.u64(d.index().get().unwrap_or(u64::MAX));
                    h.u64(d.squared_distance().to_bits());
                }
                p(&format!("{tag} discords"), h);
                let mut h = Hash::new();
                let params = MotifParams {
                    max_motifs: 3,
                    ..MotifParams::default()
                };
                for g in motifs(series, n, scale, nz(all), &params) {
                    rows(&mut h, &g);
                }
                p(&format!("{tag} motifs"), h);
                let mut h = Hash::new();
                if let Some((curve, locs)) = fluss(series, n, scale, nz(all), m, 3, 5, None) {
                    h.f64s(&curve);
                    for l in locs {
                        h.u64(l as u64);
                    }
                }
                p(&format!("{tag} fluss"), h);
                let other = ticks(walk(29, 500, 3));
                let tick = Scale::new(0.01).expect("valid scale");
                let mut h = Hash::new();
                rows(
                    &mut h,
                    &ab_join_single_threaded(series, &other, n, scale, tick),
                );
                rows(
                    &mut h,
                    &ab_join_single_threaded(&other, series, n, tick, scale),
                );
                p(&format!("{tag} ab_join"), h);
                let q = series[100..100 + m].to_vec();
                let mut h = Hash::new();
                if let Some(dp) = distance_profile(&q, series, scale) {
                    h.u64(dp.len() as u64);
                    for e in dp {
                        h.entry(e);
                    }
                }
                if let Some(ms) = find_matches(&q, series, scale, MaxDistance::Default, Some(10)) {
                    rows(&mut h, &ms);
                }
                p(&format!("{tag} pattern"), h);
                let mut lib = PatternLibrary::new(n, scale);
                let _ = lib.register(&q);
                let _ = lib.register(&series[200..200 + m]);
                let mut ps = PatternStream::new(lib);
                let mut h = Hash::new();
                for &x in series {
                    for &e in ps.push(x) {
                        h.entry(e);
                    }
                }
                p(&format!("{tag} pattern_stream"), h);

                let forces = [
                    ("natural", RouteForce::default()),
                    (
                        "wide",
                        RouteForce {
                            wide: true,
                            ..RouteForce::default()
                        },
                    ),
                    (
                        "reanchor",
                        RouteForce {
                            reanchor_every: 5,
                            anchor_seed: 9,
                            ..RouteForce::default()
                        },
                    ),
                ];
                let w = nz(64);
                for (fname, force) in forces {
                    let _g = forced_route(force);
                    let mut s = GrowingStream::new(n, scale);
                    let mut ck = Hash::new();
                    let mut floss = Floss::new(n, nz(200), usize::MAX, m, 5);
                    let mut out = Vec::new();
                    for (i, &x) in series.iter().enumerate() {
                        if let Some((r, e)) = s.update(x) {
                            out.push((r, e));
                            floss.update(r, e);
                        }
                        if i % 233 == 232 {
                            ck.bytes(&s.checkpoint());
                        }
                    }
                    let mut h = Hash::new();
                    stream_rows(&mut h, &out);
                    p(&format!("{tag} growing[{fname}]"), h);
                    routes(&format!("{tag} growing_routes[{fname}]"), s.route_counts());
                    ck.bytes(&s.checkpoint());
                    p(&format!("{tag} growing_checkpoint[{fname}]"), ck);
                    let mut h = Hash::new();
                    h.f64s(&floss.curve());
                    p(&format!("{tag} floss_growing[{fname}]"), h);
                    let mut a = GrowingStream::new(n, scale);
                    for &x in &series[..350] {
                        a.update(x);
                    }
                    let mut b = GrowingStream::restore(&a.checkpoint()).expect("restore");
                    let mut h = Hash::new();
                    for &x in &series[350..] {
                        if let Some((r, e)) = b.update(x) {
                            h.row(r.get().unwrap_or(u64::MAX), e);
                        }
                    }
                    h.bytes(&b.checkpoint());
                    p(&format!("{tag} growing_restored[{fname}]"), h);

                    let mut s = BoundedStream::new(n, w, scale).expect("w >= m");
                    let mut ck = Hash::new();
                    let mut floss = Floss::new(n, w, w.get() - m, m, 5);
                    let mut out = Vec::new();
                    for (i, &x) in series.iter().enumerate() {
                        if let Some((r, e)) = s.update(x) {
                            out.push((r, e));
                            floss.update(r, e);
                        }
                        if i % 233 == 232 {
                            ck.bytes(&s.checkpoint());
                        }
                    }
                    let mut h = Hash::new();
                    stream_rows(&mut h, &out);
                    p(&format!("{tag} bounded[{fname}]"), h);
                    routes(&format!("{tag} bounded_routes[{fname}]"), s.route_counts());
                    ck.bytes(&s.checkpoint());
                    p(&format!("{tag} bounded_checkpoint[{fname}]"), ck);
                    let mut h = Hash::new();
                    h.f64s(&floss.curve());
                    p(&format!("{tag} floss_bounded[{fname}]"), h);
                    let mut a = BoundedStream::new(n, w, scale).expect("w >= m");
                    for &x in &series[..350] {
                        a.update(x);
                    }
                    let mut b = BoundedStream::restore(&a.checkpoint()).expect("restore");
                    let mut h = Hash::new();
                    for &x in &series[350..] {
                        if let Some((r, e)) = b.update(x) {
                            h.row(r.get().unwrap_or(u64::MAX), e);
                        }
                    }
                    h.bytes(&b.checkpoint());
                    p(&format!("{tag} bounded_restored[{fname}]"), h);
                }
                znorm_routes(&tag, series, n, scale, all);
                nonnorm_batch(&tag, series, n, scale, all);
                nonnorm_streams(&tag, series, n, scale);
            }
        }
        multi(&corpus, all);
    }
}

#[cfg(not(any(loom, shuttle)))]
fn main() {
    corpus::main();
}

// The route hook it prints is compiled out where a model checker swaps the sync layer.
#[cfg(any(loom, shuttle))]
fn main() {}
