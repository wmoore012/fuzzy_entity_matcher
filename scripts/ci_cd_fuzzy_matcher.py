#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MusicScope

"""
CI/CD Integration for Fuzzy Entity Matcher

Handles the expected partial success nature of fuzzy matching in production ETL.
Designed for integration with existing YouTube ETL pipeline.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzy_entity_matcher import FuzzyMatcherBenchmark


@dataclass
class ETLQualityGate:
    """Quality gate thresholds for fuzzy matching in ETL."""

    min_artist_match_rate: float = 0.60  # 60% minimum artist match rate
    min_song_link_rate: float = 0.30  # 30% minimum song link rate
    max_error_rate: float = 0.05  # 5% maximum error rate
    min_throughput: float = 100.0  # 100 matches/second minimum
    max_memory_usage_mb: float = 500.0  # 500MB maximum memory usage


class FuzzyMatcherCICD:
    """
    CI/CD integration for fuzzy entity matcher.

    Handles expected partial success and provides quality gates for production deployment.
    """

    def __init__(self, engine=None):
        self.engine = engine
        self.logger = self._setup_logging()
        self.quality_gate = ETLQualityGate()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for CI/CD integration."""
        logger = logging.getLogger("fuzzy_matcher_cicd")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_quality_gates(self) -> Dict[str, Any]:
        """
        Run quality gates for fuzzy matching deployment.

        Returns:
            Dict with quality gate results and deployment recommendation
        """
        self.logger.info("🚦 Running Fuzzy Matcher Quality Gates")

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quality_gates": {},
            "benchmarks": {},
            "deployment_approved": False,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Run comprehensive benchmarks
            benchmark = FuzzyMatcherBenchmark()
            benchmark_results = benchmark.run_comprehensive_benchmark(
                sample_size=2000, use_real_data=True
            )

            results["benchmarks"] = self._extract_benchmark_metrics(benchmark_results)

            # Evaluate quality gates
            results["quality_gates"] = self._evaluate_quality_gates(benchmark_results)

            # Upsert to benchmarks table if engine available
            if self.engine:
                upsert_stats = benchmark.upsert_to_oss_benchmarks_table(self.engine)
                results["benchmark_upsert"] = upsert_stats
                self.logger.info(f"📊 Upserted {upsert_stats['inserted']} benchmark records")

            # Make deployment decision
            results["deployment_approved"] = self._should_approve_deployment(
                results["quality_gates"]
            )

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)

        except Exception as e:
            self.logger.error(f"❌ Quality gate execution failed: {e}")
            results["issues"].append(f"Quality gate execution failed: {str(e)}")
            results["deployment_approved"] = False

        return results

    def _extract_benchmark_metrics(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from benchmark results."""
        metrics = {
            "algorithms_tested": 0,
            "best_throughput": 0.0,
            "best_cache_hit_rate": 0.0,
            "memory_usage_mb": 0.0,
            "total_tests_run": 0,
        }

        try:
            # Extract from algorithm benchmarks
            if "algorithms" in benchmark_results:
                algorithms = benchmark_results["algorithms"]
                metrics["algorithms_tested"] = len(algorithms)

                for alg_name, result in algorithms.items():
                    if hasattr(result, "matches_per_second"):
                        metrics["best_throughput"] = max(
                            metrics["best_throughput"], result.matches_per_second
                        )
                    if hasattr(result, "cache_hit_rate"):
                        metrics["best_cache_hit_rate"] = max(
                            metrics["best_cache_hit_rate"], result.cache_hit_rate
                        )
                    if hasattr(result, "memory_usage_mb"):
                        metrics["memory_usage_mb"] = max(
                            metrics["memory_usage_mb"], result.memory_usage_mb
                        )
                    metrics["total_tests_run"] += 1

            # Extract from cache benchmarks
            if "cache" in benchmark_results:
                cache_results = benchmark_results["cache"]
                if "warm" in cache_results and hasattr(cache_results["warm"], "cache_hit_rate"):
                    metrics["best_cache_hit_rate"] = max(
                        metrics["best_cache_hit_rate"],
                        cache_results["warm"].cache_hit_rate,
                    )

        except Exception as e:
            self.logger.warning(f"⚠️  Error extracting benchmark metrics: {e}")

        return metrics

    def _evaluate_quality_gates(
        self, benchmark_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate quality gates against benchmark results."""
        gates = {}

        # Extract metrics for evaluation
        metrics = self._extract_benchmark_metrics(benchmark_results)

        # Throughput gate
        gates["throughput"] = {
            "threshold": self.quality_gate.min_throughput,
            "actual": metrics["best_throughput"],
            "passed": metrics["best_throughput"] >= self.quality_gate.min_throughput,
            "description": f"Minimum {self.quality_gate.min_throughput} matches/second required",
        }

        # Memory usage gate
        gates["memory_usage"] = {
            "threshold": self.quality_gate.max_memory_usage_mb,
            "actual": metrics["memory_usage_mb"],
            "passed": metrics["memory_usage_mb"] <= self.quality_gate.max_memory_usage_mb,
            "description": f"Maximum {self.quality_gate.max_memory_usage_mb}MB memory usage allowed",
        }

        # Cache effectiveness gate (performance indicator)
        gates["cache_effectiveness"] = {
            "threshold": 0.70,  # 70% cache hit rate minimum
            "actual": metrics["best_cache_hit_rate"],
            "passed": metrics["best_cache_hit_rate"] >= 0.70,
            "description": "Minimum 70% cache hit rate for production performance",
        }

        # Algorithm coverage gate
        gates["algorithm_coverage"] = {
            "threshold": 3,  # At least 3 algorithms tested
            "actual": metrics["algorithms_tested"],
            "passed": metrics["algorithms_tested"] >= 3,
            "description": "Minimum 3 matching algorithms must be tested",
        }

        return gates

    def _should_approve_deployment(self, quality_gates: Dict[str, Dict[str, Any]]) -> bool:
        """
        Determine if deployment should be approved.

        Uses a weighted approach - critical gates must pass, others are advisory.
        """
        critical_gates = ["throughput", "memory_usage"]
        advisory_gates = ["cache_effectiveness", "algorithm_coverage"]

        # All critical gates must pass
        critical_passed = all(
            quality_gates.get(gate, {}).get("passed", False) for gate in critical_gates
        )

        # At least 50% of advisory gates should pass
        advisory_results = [
            quality_gates.get(gate, {}).get("passed", False) for gate in advisory_gates
        ]
        advisory_pass_rate = (
            sum(advisory_results) / len(advisory_results) if advisory_results else 0
        )

        return critical_passed and advisory_pass_rate >= 0.5

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []

        quality_gates = results.get("quality_gates", {})

        # Throughput recommendations
        throughput_gate = quality_gates.get("throughput", {})
        if not throughput_gate.get("passed", False):
            recommendations.append(
                f"⚡ Improve throughput: Current {throughput_gate.get('actual', 0):.0f} matches/sec "
                f"is below minimum {throughput_gate.get('threshold', 0):.0f}. "
                f"Consider optimizing algorithm selection or increasing cache size."
            )

        # Memory recommendations
        memory_gate = quality_gates.get("memory_usage", {})
        if not memory_gate.get("passed", False):
            recommendations.append(
                f"💾 Reduce memory usage: Current {memory_gate.get('actual', 0):.1f}MB "
                f"exceeds maximum {memory_gate.get('threshold', 0):.1f}MB. "
                f"Consider reducing cache size or optimizing data structures."
            )

        # Cache recommendations
        cache_gate = quality_gates.get("cache_effectiveness", {})
        if not cache_gate.get("passed", False):
            recommendations.append(
                f"🔄 Improve cache hit rate: Current {cache_gate.get('actual', 0):.1%} "
                f"is below optimal {cache_gate.get('threshold', 0):.1%}. "
                f"Consider adjusting cache size or improving cache key strategy."
            )

        # Success recommendations
        if results.get("deployment_approved", False):
            recommendations.append(
                "✅ All critical quality gates passed. Deployment approved for production."
            )
            recommendations.append(
                "🚀 Consider monitoring cache hit rates and throughput in production."
            )

        return recommendations

    def run_etl_integration_test(self) -> Dict[str, Any]:
        """
        Run integration test simulating ETL pipeline usage.

        Tests the expected partial success scenario for YouTube video enhancement.
        """
        self.logger.info("🔧 Running ETL Integration Test")

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_type": "etl_integration",
            "success": False,
            "metrics": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # Simulate YouTube video enhancement scenario
            from fuzzy_entity_matcher import find_matches

            # Simulate database entities (what would come from your database)
            mock_artists = [
                "Taylor Swift",
                "Ed Sheeran",
                "Ariana Grande",
                "Drake",
                "The Weeknd",
                "Billie Eilish",
                "Post Malone",
                "Dua Lipa",
                "Harry Styles",
                "Olivia Rodrigo",
            ]

            mock_songs = [
                "Anti-Hero",
                "Shape of You",
                "Thank U Next",
                "Blinding Lights",
                "Bad Guy",
                "Circles",
                "Levitating",
                "Watermelon Sugar",
                "Good 4 U",
            ]

            # Simulate YouTube video titles (realistic variations)
            youtube_titles = [
                "TaylorSwiftVEVO",
                "Ed Sheeran Official",
                "ArianaGrandeVevo",
                "Drake - God's Plan (Official Music Video)",
                "The Weeknd - Blinding Lights [Official Audio]",
                "billie eilish - bad guy (slowed + reverb)",
                "Post Malone: Circles (Lyric Video)",
                "Some Random Channel - Unknown Song",
                "Dua Lipa - Levitating ft. DaBaby (Official Music Video)",
                "Harry Styles - Watermelon Sugar (Official Video)",
            ]

            # Test artist matching (expected ~70% success rate)
            artist_matches = 0
            artist_tests = 0

            for title in youtube_titles:
                artist_tests += 1
                matches = find_matches(title, mock_artists, threshold=0.7)
                if matches:
                    artist_matches += 1

            artist_match_rate = artist_matches / artist_tests if artist_tests > 0 else 0

            # Test song matching (expected ~50% success rate due to variations)
            song_matches = 0
            song_tests = 0

            for title in youtube_titles:
                song_tests += 1
                # Extract potential song name (basic extraction)
                if " - " in title:
                    potential_song = title.split(" - ", 1)[1]
                    potential_song = potential_song.replace("(Official Music Video)", "")
                    potential_song = potential_song.replace("[Official Audio]", "")
                    potential_song = potential_song.replace("(slowed + reverb)", "")
                    potential_song = potential_song.strip()

                    matches = find_matches(potential_song, mock_songs, threshold=0.8)
                    if matches:
                        song_matches += 1

            song_match_rate = song_matches / song_tests if song_tests > 0 else 0

            # Evaluate against quality gates
            results["metrics"] = {
                "artist_match_rate": artist_match_rate,
                "song_match_rate": song_match_rate,
                "artist_matches": artist_matches,
                "song_matches": song_matches,
                "total_tests": len(youtube_titles),
            }

            # Check quality gates
            artist_gate_passed = artist_match_rate >= self.quality_gate.min_artist_match_rate
            song_gate_passed = song_match_rate >= self.quality_gate.min_song_link_rate

            results["quality_gates"] = {
                "artist_matching": {
                    "passed": artist_gate_passed,
                    "threshold": self.quality_gate.min_artist_match_rate,
                    "actual": artist_match_rate,
                },
                "song_linking": {
                    "passed": song_gate_passed,
                    "threshold": self.quality_gate.min_song_link_rate,
                    "actual": song_match_rate,
                },
            }

            # Overall success (partial success is expected and acceptable)
            results["success"] = artist_gate_passed  # Song linking is optional

            # Generate recommendations
            if not artist_gate_passed:
                results["recommendations"].append(
                    f"🎤 Artist match rate {artist_match_rate:.1%} below minimum {self.quality_gate.min_artist_match_rate:.1%}. "
                    f"Consider lowering threshold or improving preprocessing."
                )

            if not song_gate_passed:
                results["recommendations"].append(
                    f"🎵 Song link rate {song_match_rate:.1%} below target {self.quality_gate.min_song_link_rate:.1%}. "
                    f"This is expected for YouTube data with many variations."
                )

            if results["success"]:
                results["recommendations"].append(
                    "✅ ETL integration test passed. Ready for production YouTube enhancement."
                )

        except Exception as e:
            self.logger.error(f"❌ ETL integration test failed: {e}")
            results["issues"].append(f"ETL integration test failed: {str(e)}")
            results["success"] = False

        return results

    def generate_ci_cd_report(self) -> Dict[str, Any]:
        """Generate comprehensive CI/CD report."""
        self.logger.info("📋 Generating CI/CD Report")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "fuzzy-entity-matcher",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "quality_gates": {},
            "etl_integration": {},
            "deployment_decision": {},
            "next_steps": [],
        }

        try:
            # Run quality gates
            quality_results = self.run_quality_gates()
            report["quality_gates"] = quality_results

            # Run ETL integration test
            etl_results = self.run_etl_integration_test()
            report["etl_integration"] = etl_results

            # Make deployment decision
            quality_approved = quality_results.get("deployment_approved", False)
            etl_approved = etl_results.get("success", False)

            report["deployment_decision"] = {
                "approved": quality_approved and etl_approved,
                "quality_gates_passed": quality_approved,
                "etl_integration_passed": etl_approved,
                "reasoning": self._get_deployment_reasoning(quality_approved, etl_approved),
            }

            # Generate next steps
            if report["deployment_decision"]["approved"]:
                report["next_steps"] = [
                    "✅ Deploy to production",
                    "📊 Monitor performance metrics",
                    "🔄 Set up automated benchmarking",
                    "📈 Track YouTube enhancement success rates",
                ]
            else:
                report["next_steps"] = [
                    "❌ Fix failing quality gates",
                    "🔧 Optimize performance issues",
                    "🧪 Re-run tests after fixes",
                    "📋 Review recommendations",
                ]

        except Exception as e:
            self.logger.error(f"❌ CI/CD report generation failed: {e}")
            report["deployment_decision"] = {
                "approved": False,
                "reasoning": f"Report generation failed: {str(e)}",
            }

        return report

    def _get_deployment_reasoning(self, quality_approved: bool, etl_approved: bool) -> str:
        """Get human-readable deployment reasoning."""
        if quality_approved and etl_approved:
            return "All quality gates and ETL integration tests passed. Safe for production deployment."
        elif quality_approved and not etl_approved:
            return "Quality gates passed but ETL integration has issues. Review ETL test results."
        elif not quality_approved and etl_approved:
            return (
                "ETL integration passed but quality gates failed. Performance optimization needed."
            )
        else:
            return "Both quality gates and ETL integration failed. Significant fixes required."


def main():
    """Main entry point for CI/CD integration."""
    print("🚀 Fuzzy Entity Matcher CI/CD Integration")
    print("=" * 50)

    # Initialize CI/CD integration
    try:
        # Try to connect to database for benchmark upserts
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "web"))
            from db_guard import get_engine

            engine = get_engine("PUBLIC", ro=False)
            print("✅ Database connection established for benchmark upserts")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            print("📊 Benchmarks will be saved to JSON only")
            engine = None

        cicd = FuzzyMatcherCICD(engine=engine)

        # Generate comprehensive report
        report = cicd.generate_ci_cd_report()

        # Save report
        report_file = f"fuzzy_matcher_cicd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📋 CI/CD Report saved to: {report_file}")

        # Print summary
        print("\n📊 CI/CD Summary:")
        print("-" * 30)

        deployment_approved = report["deployment_decision"]["approved"]
        print(f"🚦 Deployment Approved: {'✅ YES' if deployment_approved else '❌ NO'}")

        if "quality_gates" in report and "quality_gates" in report["quality_gates"]:
            quality_gates = report["quality_gates"]["quality_gates"]
            passed_gates = sum(1 for gate in quality_gates.values() if gate.get("passed", False))
            total_gates = len(quality_gates)
            print(f"🎯 Quality Gates: {passed_gates}/{total_gates} passed")

        if "etl_integration" in report and "metrics" in report["etl_integration"]:
            metrics = report["etl_integration"]["metrics"]
            artist_rate = metrics.get("artist_match_rate", 0)
            song_rate = metrics.get("song_match_rate", 0)
            print(f"🎤 Artist Match Rate: {artist_rate:.1%}")
            print(f"🎵 Song Link Rate: {song_rate:.1%}")

        # Print next steps
        if "next_steps" in report:
            print("\n📋 Next Steps:")
            for step in report["next_steps"]:
                print(f"   {step}")

        # Exit with appropriate code for CI/CD
        exit_code = 0 if deployment_approved else 1
        print(f"\n🏁 Exiting with code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ CI/CD integration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
