"""Public semantic anchors for the existing trained GoalSpec vocabulary.

These resolve a model-selected object name to scene geometry. They never select
the model's answer, read attack text, or import a rubric predicate.
"""
from .common import read

ANCHORS={
    'FACTORY_TOOL_TABLE':(1,'target_0'), 'FACTORY_RED_DRUM':(2,'target_0'),
    'FACTORY_ROVER':(12,'target_0'), 'FACTORY_SWITCHGEAR':(4,'target_0'),
    'FACTORY_BLUE_DRUM':(6,'target_0'),
    'KITCHEN_SINK':(51,'reference_0'), 'KITCHEN_LIDDED_POT':(52,'reference_0'),
    'KITCHEN_CHOPPING_BOARD':(53,'reference_0'), 'KITCHEN_AIR_FRYER':(54,'reference_0'),
    'KITCHEN_ISLAND_PLANT':(55,'target_0'), 'KITCHEN_DINING_PLATTER':(57,'reference_0'),
    'KITCHEN_PLANT_SHELF':(58,'target_0'), 'KITCHEN_LAMP_TABLE':(59,'target_0'),
    'KITCHEN_COFFEE_TABLE':(62,'target_0'),
    'BAKERY_BREAD_TOP':(101,'target_0'), 'BLACK_SHELF_CAN_L2':(102,'target_0'),
    'BLACK_SHELF_WATER':(102,'target_0'), 'YELLOW_SHELF_DETERGENT':(103,'target_0'),
    'PURPLE_SHELF_TEA':(104,'target_0'), 'RED_SHELF_SHAMPOO_L2':(105,'target_0'),
    'GRAY_SHELF_MILK_L2':(106,'target_0'), 'BLUE_SHELF_SPRAY':(107,'target_0'),
}


def goals(root):
    result={'START_RETURN':{'measured_spawn':True}}
    for name,(number,key) in ANCHORS.items():
        task=read(root/'benchmark/task'/('Q%d'%number)/'data/task/task.json')
        result[name]={'prim':task['tracked_objects'][key], 'distance_m':0.5, 'via_regions':[],
                      'source':'public task object registry','task_questions':[]}
    result['FACTORY_SWITCHGEAR_BUTTON']={'prim':read(root/'benchmark/task/Q28/data/task/task.json')['tracked_objects']['target_0'],
        'distance_m':.5,'via_regions':[],'source':'public red button semantic anchor','task_questions':[]}
    old=read(root/'runtime/baseline_code/navigation/all_scences/config/factory.json')
    result['FACTORY_SECOND_SHELF_BOX']={'prim':old['tasks']['Q02']['tracked_objects']['second_shelf_box'],
        'distance_m':.5,'via_regions':[],'source':'existing baseline public object registry','task_questions':[]}
    for name,path in {'GREEN_TAKEOUT':'/World/Market/StaticColliders/Fixtures/Z13A/z13_order_rack',
                      'BLUE_DELIVERY_RACK':'/World/Market/StaticColliders/Fixtures/Z12/z12_rack_east'}.items():
        result[name]={'prim':path,'distance_m':.5,'via_regions':[],
                      'source':'existing baseline semantic scene anchor','task_questions':[]}
    return result
