; Annotation declarations
(annotation_type_declaration
  body: (_)) @annotation.declaration

; Class declarations
(class_declaration
  body: (_)) @class.declaration

; Interface declarations
(interface_declaration
  body: (_)) @interface.declaration

; Method declarations
(method_declaration
  body: (_)?) @method.declaration

; Field declarations
(field_declaration) @field.declaration

; Enum declarations
(enum_declaration
  body: (enum_body
    (enum_constant)*)) @enum.declaration

; Annotations to strip
(annotation) @annotation
