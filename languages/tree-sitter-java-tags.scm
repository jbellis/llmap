(class_declaration
  (modifiers)? @class.modifiers
  name: (identifier) @name.definition.class
  superclass: (superclass)? @class.superclass
  interfaces: (super_interfaces)? @class.interfaces
  body: (class_body) @class.body) @definition.class

(method_declaration
  (modifiers)? @method.modifiers
  type: (_) @method.type
  name: (identifier) @name.definition.method
  parameters: (formal_parameters) @method.params
  body: (_)?) @definition.method

(interface_declaration
  (modifiers)? @interface.modifiers
  name: (identifier) @name.definition.interface
  interfaces: (extends_interfaces)? @interface.extends) @definition.interface

(field_declaration
  type: (_)
  declarator: (variable_declarator)) @definition.field
