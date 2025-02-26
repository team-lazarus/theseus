from theseus.utils.actors import Player, Enemy, Bullet
from theseus.utils.state import State

def test_state_initialization():
    """Test that State is initialized correctly with valid inputs."""
    # Arrange
    hero_data = {
        "position": [100, 200],
        "health": 10,
        "phase_cooldown": 5.0,
        "ability_cooldown": 10.0,
        "shoot_cooldown": 0.5
    }
    
    bullets_data = [
        {"position": [50, 60], "direction": 1.5, "type": "standard"},
        {"position": [70, 80], "direction": 0.8, "type": "special"}
    ]
    
    enemies_data = [
        {"position": [300, 400], "health": 5, "direction": 2.0, "type": "basic"},
        {"position": [500, 600], "health": 8, "direction": 3.0, "type": "advanced"}
    ]
    
    doors = [[700, 100, 750, 150]]
    backdoor = [50, 50, 100, 100]
    walls = [[0, 0, 800, 20], [0, 0, 20, 600]]
    
    # Act
    state = State(hero_data, bullets_data, enemies_data, doors, backdoor, walls)
    
    # Assert
    assert isinstance(state.hero, Player)
    assert state.hero.x == 100
    assert state.hero.y == 200
    assert state.hero.health == 10
    
    bullets_list = list(state.bullets)
    assert all(isinstance(bullet, Bullet) for bullet in bullets_list)
    assert state.doors == doors
    assert state.backdoor == backdoor
    assert state.walls == walls

def test_empty_collections():
    """Test initialization with empty collections."""
    # Arrange
    hero_data = {"position": [100, 200], "health": 10, "phase_cooldown": 0.0, 
                 "ability_cooldown": 0.0, "shoot_cooldown": 0.0}
    
    # Act
    state = State(hero_data, [], [], [], [], [])
    
    # Assert
    assert isinstance(state.hero, Player)
    assert list(state.bullets) == []
    assert list(state.enemies) == []
    assert state.doors == []
    assert state.backdoor == []
    assert state.walls == []

def test_bullet_conversion():
    """Test that bullet dictionaries are properly converted to Bullet objects."""
    # Arrange
    hero_data = {"position": [0, 0], "health": 1, "phase_cooldown": 0.0, 
                 "ability_cooldown": 0.0, "shoot_cooldown": 0.0}
    
    bullets_data = [
        {"position": [10, 20], "direction": 1.0, "type": "standard"},
        {"position": [30, 40], "direction": 2.0, "type": "piercing"}
    ]
    
    # Act
    state = State(hero_data, bullets_data, [], [], [], [])
    bullets_list = list(state.bullets)
    
    # Assert
    assert len(bullets_list) == 2
    assert all(isinstance(bullet, Bullet) for bullet in bullets_list)
    assert bullets_list[0].x == 10
    assert bullets_list[0].y == 20
    assert bullets_list[0].direction == 1.0
    assert bullets_list[0].type_ == "standard"
    assert bullets_list[1].x == 30
    assert bullets_list[1].y == 40

def test_enemy_conversion():
    """Test that enemy dictionaries are properly converted to Enemy objects."""
    # Arrange
    hero_data = {"position": [0, 0], "health": 1, "phase_cooldown": 0.0, 
                 "ability_cooldown": 0.0, "shoot_cooldown": 0.0}
    
    enemies_data = [
        {"position": [100, 200], "health": 3, "direction": 0.5, "type": "grunt"},
        {"position": [300, 400], "health": 5, "direction": 1.5, "type": "boss"}
    ]
    
    # Act
    state = State(hero_data, [], enemies_data, [], [], [])
    enemies_list = list(state.enemies)
    
    # Assert
    assert len(enemies_list) == 2
    assert all(isinstance(enemy, Enemy) for enemy in enemies_list)
    assert enemies_list[0].x == 100
    assert enemies_list[0].y == 200
    assert enemies_list[0].health == 3
    assert enemies_list[0].type_ == "grunt"
    assert enemies_list[1].x == 300
    assert enemies_list[1].y == 400

def test_bullet_on_target():
    """Test the Bullet.on_target method."""
    # Arrange
    hero_data = {"position": [0, 0], "health": 1, "phase_cooldown": 0.0, 
                 "ability_cooldown": 0.0, "shoot_cooldown": 0.0}
    
    # A bullet at (10, 10) pointing at (20, 20) - 45 degree angle
    bullet_data = [{"position": [10, 10], "direction": 0.7853981634, "type": "standard"}]  # pi/4 radians
    
    # Act
    state = State(hero_data, bullet_data, [], [], [], [])
    bullet = list(state.bullets)[0]
    
    # Assert
    assert bullet.on_target([20, 20])  # Should return True for target in the bullet's path
    assert not bullet.on_target([20, 10])  # Should return False for target not in path
