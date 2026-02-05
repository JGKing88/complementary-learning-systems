"""Tests for generate_shell_displacements function."""

import pytest


def generate_shell_displacements(min_step, max_step):
    """Generate (dx, dy) displacement pairs for a shell [min_step, max_step).
    
    A "shell" consists of all displacements where min_step <= max(|dx|, |dy|) < max_step.
    This traces the perimeter of square rings without filtering.
    """
    for m in range(min_step, max_step):
        if m == 0:
            continue
        # Top edge: dy = m, dx from -m to m
        for dx in range(-m, m + 1):
            yield (dx, m)
        # Right edge: dx = m, dy from m-1 down to -m (excluding top corner)
        for dy in range(m - 1, -m - 1, -1):
            yield (m, dy)
        # Bottom edge: dy = -m, dx from m-1 down to -m (excluding right corner)
        for dx in range(m - 1, -m - 1, -1):
            yield (dx, -m)
        # Left edge: dx = -m, dy from -m+1 to m-1 (excluding corners)
        for dy in range(-m + 1, m):
            yield (-m, dy)


def displacement_to_label(dx, dy):
    """Convert a displacement (dx, dy) to an action label."""
    if dx == 0 and dy == 0:
        raise ValueError("Zero displacement has no direction")
    
    abs_dx, abs_dy = abs(dx), abs(dy)
    
    if abs_dx > abs_dy:
        return 1 if dx > 0 else 3  # East or West
    elif abs_dy > abs_dx:
        return 0 if dy > 0 else 2  # North or South
    else:
        # Tie: clockwise priority E > S > W > N
        if dx > 0:
            return 1  # East
        elif dy < 0:
            return 2  # South
        elif dx < 0:
            return 3  # West
        else:
            return 0  # North


class TestGenerateShellDisplacements:
    """Test suite for generate_shell_displacements."""
    
    def test_single_step_shell(self):
        """Shell [1, 2) should produce exactly the 8 single-step moves."""
        displacements = list(generate_shell_displacements(1, 2))
        
        # Should have exactly 8 displacements (the ring at magnitude 1)
        assert len(displacements) == 8
        
        # All should have max magnitude of 1
        for dx, dy in displacements:
            assert max(abs(dx), abs(dy)) == 1
        
        # Should include all 8 directions
        expected = {(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)}
        assert set(displacements) == expected
    
    def test_magnitude_two_ring(self):
        """Shell [2, 3) should produce the ring at magnitude 2."""
        displacements = list(generate_shell_displacements(2, 3))
        
        # Ring at magnitude 2 has 4 sides × 5 positions - 4 corners counted once = 16
        # Actually: perimeter = 8 * m = 16 for m=2
        assert len(displacements) == 16
        
        # All should have max magnitude of exactly 2
        for dx, dy in displacements:
            assert max(abs(dx), abs(dy)) == 2
        
        # Check some specific points
        assert (0, 2) in displacements  # Top center
        assert (2, 0) in displacements  # Right center
        assert (2, 2) in displacements  # Top-right corner
        assert (-2, -2) in displacements  # Bottom-left corner
    
    def test_multi_ring_shell(self):
        """Shell [1, 3) should combine rings at magnitude 1 and 2."""
        displacements = list(generate_shell_displacements(1, 3))
        
        # Ring 1: 8 points, Ring 2: 16 points = 24 total
        assert len(displacements) == 8 + 16
        
        # All should have max magnitude between 1 and 2 inclusive
        for dx, dy in displacements:
            mag = max(abs(dx), abs(dy))
            assert 1 <= mag <= 2
    
    def test_shell_excludes_inner(self):
        """Shell [2, 4) should NOT include magnitude 1."""
        displacements = list(generate_shell_displacements(2, 4))
        
        for dx, dy in displacements:
            mag = max(abs(dx), abs(dy))
            assert mag >= 2, f"Found magnitude {mag} < 2 for ({dx}, {dy})"
            assert mag < 4, f"Found magnitude {mag} >= 4 for ({dx}, {dy})"
    
    def test_shell_excludes_outer(self):
        """Shell [1, 3) should NOT include magnitude 3 or higher."""
        displacements = list(generate_shell_displacements(1, 3))
        
        for dx, dy in displacements:
            mag = max(abs(dx), abs(dy))
            assert mag < 3, f"Found magnitude {mag} >= 3 for ({dx}, {dy})"
    
    def test_empty_shell(self):
        """Shell [2, 2) should be empty (min == max)."""
        displacements = list(generate_shell_displacements(2, 2))
        assert len(displacements) == 0
    
    def test_zero_excluded(self):
        """Zero displacement should never be generated."""
        for min_s in range(0, 3):
            for max_s in range(min_s + 1, 5):
                displacements = list(generate_shell_displacements(min_s, max_s))
                assert (0, 0) not in displacements
    
    def test_no_duplicates(self):
        """Shell should not contain duplicate displacements."""
        for min_s in range(1, 4):
            for max_s in range(min_s + 1, 6):
                displacements = list(generate_shell_displacements(min_s, max_s))
                assert len(displacements) == len(set(displacements)), \
                    f"Duplicates found in shell [{min_s}, {max_s})"
    
    def test_ring_size_formula(self):
        """Each ring at magnitude m should have exactly 8*m points."""
        for m in range(1, 6):
            ring = list(generate_shell_displacements(m, m + 1))
            expected_size = 8 * m
            assert len(ring) == expected_size, \
                f"Ring at m={m}: expected {expected_size}, got {len(ring)}"
    
    def test_symmetry(self):
        """Shell should be symmetric under 90-degree rotations."""
        displacements = set(generate_shell_displacements(1, 4))
        
        for dx, dy in list(displacements):
            # 90° rotation: (dx, dy) -> (-dy, dx)
            assert (-dy, dx) in displacements, f"Missing 90° rotation of ({dx}, {dy})"
            # 180° rotation: (dx, dy) -> (-dx, -dy)
            assert (-dx, -dy) in displacements, f"Missing 180° rotation of ({dx}, {dy})"
            # 270° rotation: (dx, dy) -> (dy, -dx)
            assert (dy, -dx) in displacements, f"Missing 270° rotation of ({dx}, {dy})"


class TestDisplacementToLabel:
    """Test displacement_to_label function."""
    
    def test_cardinal_directions(self):
        """Test pure cardinal directions."""
        assert displacement_to_label(0, 1) == 0  # North
        assert displacement_to_label(1, 0) == 1  # East
        assert displacement_to_label(0, -1) == 2  # South
        assert displacement_to_label(-1, 0) == 3  # West
    
    def test_larger_magnitude_wins(self):
        """Larger magnitude axis should determine direction."""
        assert displacement_to_label(3, 1) == 1  # East (|3| > |1|)
        assert displacement_to_label(-3, 1) == 3  # West (|-3| > |1|)
        assert displacement_to_label(1, 3) == 0  # North (|3| > |1|)
        assert displacement_to_label(1, -3) == 2  # South (|-3| > |1|)
    
    def test_tie_clockwise_priority(self):
        """Ties should be broken by clockwise priority: E > S > W > N."""
        # (+, +) -> East wins over North
        assert displacement_to_label(2, 2) == 1  # East
        # (+, -) -> East wins over South
        assert displacement_to_label(2, -2) == 1  # East
        # (-, -) -> South wins over West
        assert displacement_to_label(-2, -2) == 2  # South
        # (-, +) -> West wins over North
        assert displacement_to_label(-2, 2) == 3  # West
    
    def test_zero_raises(self):
        """Zero displacement should raise ValueError."""
        with pytest.raises(ValueError):
            displacement_to_label(0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

