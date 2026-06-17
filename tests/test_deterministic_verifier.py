from __future__ import annotations

import unittest

from agent.deterministic_verifier import verify_deterministic
from agent.execution import ExecutionResult, execute_sql


class DeterministicVerifierTests(unittest.TestCase):
    def issue(
        self,
        db_id: str,
        question: str,
        sql: str,
        execution: ExecutionResult | None = None,
    ) -> str | None:
        if execution is None:
            execution = execute_sql(db_id, sql)
        return verify_deterministic(db_id, question, sql, execution)

    def test_rejects_multiple_statements(self) -> None:
        issue = self.issue("formula_1", "List races.", "SELECT 1; SELECT 2;")
        self.assertIn("exactly one statement", issue or "")

    def test_detects_duplicate_rows_without_distinct(self) -> None:
        sql = (
            "SELECT c.lat, c.lng "
            "FROM circuits c JOIN races r ON c.circuitId = r.circuitId "
            "WHERE r.name = 'Australian Grand Prix';"
        )
        issue = self.issue(
            "formula_1",
            "What is the coordinates location of the circuits for Australian grand prix?",
            sql,
        )
        self.assertIn("duplicate", issue or "")

    def test_detects_wrong_nces_school_identifier(self) -> None:
        sql = (
            'SELECT s."NCESDist" FROM "schools" s '
            'JOIN "frpm" f ON s."CDSCode" = f."CDSCode" '
            'ORDER BY f."Enrollment (Ages 5-17)" DESC LIMIT 5;'
        )
        issue = self.issue(
            "california_schools",
            "Please give their NCES school identification number.",
            sql,
        )
        self.assertIn("NCESSchool", issue or "")

    def test_detects_case_sensitive_literal_mismatch(self) -> None:
        sql = (
            'SELECT c."id" FROM "cards" c '
            'JOIN "legalities" l ON c."uuid" = l."uuid" '
            'WHERE c."rarity" = \'mythic\' AND l."format" = \'gladiator\' '
            "AND l.\"status\" = 'banned';"
        )
        issue = self.issue(
            "card_games",
            "List all the mythic rarity print cards banned in gladiator format.",
            sql,
        )
        self.assertIn("Banned", issue or "")

    def test_detects_missing_timestamp_fraction(self) -> None:
        sql = (
            "SELECT u.Reputation FROM users u JOIN badges b ON u.Id = b.UserId "
            "WHERE b.Date = '2010-07-19 19:39:08';"
        )
        issue = self.issue(
            "codebase_community",
            "Mention the reputation of users who had obtained the badge on 7/19/2010 7:39:08 PM.",
            sql,
        )
        self.assertIn(".0", issue or "")

    def test_detects_bad_fastest_lap_conversion(self) -> None:
        sql = (
            "SELECT AVG(CAST(REPLACE(REPLACE(r.fastestLapTime, 'm', '.'), 's', '') AS REAL)) "
            "FROM results r JOIN drivers d ON r.driverId = d.driverId "
            "WHERE d.forename = 'Lewis' AND d.surname = 'Hamilton';"
        )
        issue = self.issue(
            "formula_1",
            "What is the average fastest lap time in seconds for Lewis Hamilton in all the Formula_1 races?",
            sql,
        )
        self.assertIn("mm:ss.xxx", issue or "")

    def test_detects_concatenated_full_name(self) -> None:
        sql = (
            "SELECT m.first_name || ' ' || m.last_name AS full_name "
            "FROM member m JOIN major maj ON m.link_to_major = maj.major_id "
            "WHERE maj.department = 'Art and Design';"
        )
        issue = self.issue(
            "student_club",
            "Please list the full names of the students in the Student_Club that come from the Art and Design Department.",
            sql,
        )
        self.assertIn("separate columns", issue or "")

    def test_detects_wrong_excellence_rate_denominator(self) -> None:
        sql = (
            'SELECT s."Street", s."City", s."Zip", s."State" '
            'FROM "schools" s JOIN "satscores" ss ON s."CDSCode" = ss."cds" '
            'ORDER BY (ss."NumGE1500" * 1.0 / ss."enroll12") ASC LIMIT 1;'
        )
        issue = self.issue(
            "california_schools",
            "What is the complete address of the school with the lowest excellence rate?",
            sql,
        )
        self.assertIn("NumTstTakr", issue or "")

    def test_detects_wrong_popularity_aggregation(self) -> None:
        sql = (
            "SELECT p.Title, p.Score, p.ViewCount "
            "FROM posts p JOIN users u ON p.OwnerUserId = u.Id "
            "WHERE u.DisplayName IN ('Harvey Motulsky', 'Noah Snyder') "
            "ORDER BY p.ViewCount DESC, p.Score DESC LIMIT 1;"
        )
        issue = self.issue(
            "codebase_community",
            "Among posts by Harvey Motulsky and Noah Snyder, which one has higher popularity?",
            sql,
        )
        self.assertIn("SUM", issue or "")


if __name__ == "__main__":
    unittest.main()
