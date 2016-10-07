from rope.base.project import Project
project = Project('.')

mod1 = project.root.create_file('mod1.py')
mod1.write('def pow(x, y):\n    result = 1\n'
        '    for i in range(y):\n        result *= x\n'
        '    return result\n')
mod2 = project.root.create_file('mod2.py')
mod2.write('import mod1\nprint(mod1.pow(2, 3))\n')

from rope.refactor import restructure

pattern = '${pow_func}(${param1}, ${param2})'
goal = '${param1} ** ${param2}'
args = {'pow_func': 'name=mod1.pow'}
"""
   Example #1::

      pattern ${pyobject}.get_attribute(${name})
      goal ${pyobject}[${name}]
      args pyobject: instance=rope.base.pyobjects.PyObject

    Example #2::

      pattern ${name} in ${pyobject}.get_attributes()
      goal ${name} in {pyobject}
      args pyobject: instance=rope.base.pyobjects.PyObject

    Example #3::

      pattern ${pycore}.create_module(${project}.root, ${name})
      goal generate.create_module(${project}, ${name})

      imports
       from rope.contrib import generate

      args
       project: type=rope.base.project.Project

    Example #4::

      pattern ${pow}(${param1}, ${param2})
      goal ${param1} ** ${param2}
      args pow: name=mod.pow, exact

    Example #5::

      pattern ${inst}.longtask(${p1}, ${p2})
      goal
       ${inst}.subtask1(${p1})
       ${inst}.subtask2(${p2})
      args
       inst: type=mod.A,unsure

"""
restructuring = restructure.Restructure(project, pattern, goal, args)

project.do(restructuring.get_changes())
mod2.read()
u'import mod1\nprint(2 ** 3)\n'

# Cleaning up
mod1.remove()
mod2.remove()
project.close()