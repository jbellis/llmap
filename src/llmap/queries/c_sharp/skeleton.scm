; Using directives
(using_directive) @using.directive

; Classes
(class_declaration
  name: (identifier) @class.name) @class.declaration

; Interfaces
(interface_declaration
  name: (identifier) @interface.name) @interface.declaration

; Methods
(method_declaration
  name: (identifier) @method.name) @method.declaration

; Fields
(field_declaration
  (variable_declaration
    (variable_declarator
      name: (identifier) @field.name))) @field.declaration

; Properties
(property_declaration
  name: (identifier) @property.name) @property.declaration

; Constructor
(constructor_declaration) @constructor.declaration

; Attributes
(attribute_list) @annotation
