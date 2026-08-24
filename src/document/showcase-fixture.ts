export interface DocumentShowcaseFixtureBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DocumentShowcaseFixtureSource {
  publisher: string;
  url: string;
  retrievedAt: string;
  sha256: string;
  pdfPageCount: number;
  selectedPageNumbers: number[];
  rightsNote: string;
}

export interface DocumentShowcaseFixtureRegion {
  id: string;
  pageId: string;
  pageNumber: number;
  order: number;
  label: string;
  kind: 'title' | 'heading' | 'paragraph' | 'status' | 'table_row';
  text: string;
  bbox: DocumentShowcaseFixtureBoundingBox;
}

export interface DocumentShowcaseFixturePage {
  id: string;
  pageNumber: number;
  width: number;
  height: number;
  label: string;
  imageUrl: string;
  imageSha256: string;
  regions: DocumentShowcaseFixtureRegion[];
}

export interface DocumentShowcaseFixtureClaim {
  id: string;
  kind: 'accepted' | 'proposed';
  clause: string;
  summary: string;
  pageId: string;
  pageNumber: number;
  regionId: string;
  sourceText: string;
}

export interface DocumentShowcaseFixtureRule {
  id: string;
  clause: string;
  summary: string;
  sourceText: string;
}

export interface DocumentShowcaseFixtureQuestion {
  id: string;
  label: string;
  question: string;
  query: string;
  supportedAnswer?: string;
  unsupportedAnswer: string;
  relatedEvidenceIds: string[];
  expectedStatus: 'answered' | 'unsupported';
  expectedSourceRegionIds?: string[];
  expectedAcceptedClaimIds?: string[];
  expectedRuleIds?: string[];
}

export interface DocumentShowcaseFixture {
  id: string;
  kindLabel: string;
  fixtureVersion: string;
  fileName: string;
  title: string;
  parserMode: 'source_text_reviewed';
  parserLabel: string;
  adapterLabel: string;
  source: DocumentShowcaseFixtureSource;
  defaultQuestionId: string;
  pages: DocumentShowcaseFixturePage[];
  claims: DocumentShowcaseFixtureClaim[];
  rules: DocumentShowcaseFixtureRule[];
  questions: DocumentShowcaseFixtureQuestion[];
}

const RETRIEVED_AT = '2026-08-20T00:00:00.000Z';
const PARSER_LABEL = 'PDF text-layer extraction · human reviewed';
const ADAPTER_LABEL = 'Real page prepared for Unlimited-OCR; provider run is quota-blocked';

const IRS_W9_ENGLISH: DocumentShowcaseFixture = {
  id: 'irs-w9-english',
  kindLabel: 'Government form',
  fixtureVersion: 'irs-w9-en-2024-live-v1',
  fileName: 'irs-form-w9-en.pdf',
  title: 'IRS Form W-9 · English',
  parserMode: 'source_text_reviewed',
  parserLabel: PARSER_LABEL,
  adapterLabel: ADAPTER_LABEL,
  source: {
    publisher: 'Internal Revenue Service',
    url: 'https://www.irs.gov/pub/irs-pdf/fw9.pdf',
    retrievedAt: RETRIEVED_AT,
    sha256: '2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53',
    pdfPageCount: 6,
    selectedPageNumbers: [1],
    rightsNote: 'Official public IRS form; retained locally for source-attributed evaluation.',
  },
  defaultQuestionId: 'w9-destination',
  pages: [
    {
      id: 'w9-en-page-1',
      pageNumber: 1,
      width: 1275,
      height: 1650,
      label: 'PDF page 1',
      imageUrl: '/documents/irs-form-w9-en-p1.png',
      imageSha256: '05ebe549fcd6805423e1757b45b9901bd3d14ccd5d80281c06ce249f05948dda',
      regions: [
        {
          id: 'w9-en-region-title', pageId: 'w9-en-page-1', pageNumber: 1, order: 1,
          label: 'Region 1', kind: 'title',
          text: 'Form W-9 (Rev. March 2024) — Request for Taxpayer Identification Number and Certification',
          bbox: { x: 70, y: 68, width: 875, height: 112 },
        },
        {
          id: 'w9-en-region-destination', pageId: 'w9-en-page-1', pageNumber: 1, order: 2,
          label: 'Region 2', kind: 'status',
          text: 'Give form to the requester. Do not send to the IRS.',
          bbox: { x: 1018, y: 72, width: 188, height: 95 },
        },
        {
          id: 'w9-en-region-line-one', pageId: 'w9-en-page-1', pageNumber: 1, order: 3,
          label: 'Region 3', kind: 'paragraph',
          text: '1 Name of entity/individual. An entry is required.',
          bbox: { x: 118, y: 202, width: 1090, height: 72 },
        },
        {
          id: 'w9-en-region-tin', pageId: 'w9-en-page-1', pageNumber: 1, order: 4,
          label: 'Region 4', kind: 'paragraph',
          text: 'The TIN provided must match the name given on line 1 to avoid backup withholding.',
          bbox: { x: 74, y: 751, width: 790, height: 108 },
        },
        {
          id: 'w9-en-region-certification', pageId: 'w9-en-page-1', pageNumber: 1, order: 5,
          label: 'Region 5', kind: 'paragraph',
          text: 'Under penalties of perjury, I certify that the number shown on this form is my correct taxpayer identification number.',
          bbox: { x: 72, y: 922, width: 1134, height: 148 },
        },
        {
          id: 'w9-en-region-purpose', pageId: 'w9-en-page-1', pageNumber: 1, order: 6,
          label: 'Region 6', kind: 'paragraph',
          text: 'An individual or entity (Form W-9 requester) who is required to file an information return with the IRS is giving you this form.',
          bbox: { x: 658, y: 1471, width: 544, height: 78 },
        },
      ],
    },
  ],
  claims: [
    {
      id: 'w9-en-claim-recipient', kind: 'accepted', clause: 'form_recipient(w9_en, requester).',
      summary: 'A reviewer accepted the form recipient stated in the extracted source block.',
      pageId: 'w9-en-page-1', pageNumber: 1, regionId: 'w9-en-region-destination',
      sourceText: 'Give form to the requester. Do not send to the IRS.',
    },
    {
      id: 'w9-en-claim-prohibited-recipient', kind: 'accepted', clause: 'prohibited_recipient(w9_en, irs).',
      summary: 'A reviewer accepted the explicit prohibition on sending this form to the IRS.',
      pageId: 'w9-en-page-1', pageNumber: 1, regionId: 'w9-en-region-destination',
      sourceText: 'Give form to the requester. Do not send to the IRS.',
    },
    {
      id: 'w9-en-claim-required-line', kind: 'accepted', clause: 'required_entry(w9_en, line_1).',
      summary: 'Line 1 is explicitly marked as required.',
      pageId: 'w9-en-page-1', pageNumber: 1, regionId: 'w9-en-region-line-one',
      sourceText: '1 Name of entity/individual. An entry is required.',
    },
    {
      id: 'w9-en-claim-tin-match', kind: 'accepted', clause: 'tin_matches(w9_en, line_1_name).',
      summary: 'The TIN must match the name supplied on line 1.',
      pageId: 'w9-en-page-1', pageNumber: 1, regionId: 'w9-en-region-tin',
      sourceText: 'The TIN provided must match the name given on line 1 to avoid backup withholding.',
    },
    {
      id: 'w9-en-claim-filing-party', kind: 'proposed', clause: 'filing_party(w9_en, requester).',
      summary: 'The visible purpose paragraph suggests the requester files an information return, but the extraction remains proposed.',
      pageId: 'w9-en-page-1', pageNumber: 1, regionId: 'w9-en-region-purpose',
      sourceText: 'An individual or entity (Form W-9 requester) who is required to file an information return with the IRS is giving you this form.',
    },
  ],
  rules: [
    {
      id: 'w9-en-rule-destination',
      clause: 'safe_submission(Form, Recipient) :- form_recipient(Form, Recipient), prohibited_recipient(Form, irs).',
      summary: 'A safe submission destination requires both the named recipient and the explicit IRS prohibition.',
      sourceText: 'Reviewed form-routing rule.',
    },
    {
      id: 'w9-en-rule-identity-match',
      clause: 'identity_match_required(Form) :- required_entry(Form, line_1), tin_matches(Form, line_1_name).',
      summary: 'The required line-1 name and TIN matching statement jointly establish the identity-match requirement.',
      sourceText: 'Reviewed form-completion rule.',
    },
  ],
  questions: [
    {
      id: 'w9-destination', label: 'Where does it go?', question: 'Where should the completed W-9 go?',
      query: 'safe_submission(w9_en, Recipient)',
      supportedAnswer: 'Give the completed W-9 to the requester; do not send it to the IRS.',
      unsupportedAnswer: 'The reviewed evidence does not establish a safe submission destination.',
      relatedEvidenceIds: ['w9-en-claim-recipient', 'w9-en-claim-prohibited-recipient'],
      expectedStatus: 'answered', expectedSourceRegionIds: ['w9-en-region-destination'],
      expectedAcceptedClaimIds: ['w9-en-claim-recipient', 'w9-en-claim-prohibited-recipient'],
      expectedRuleIds: ['w9-en-rule-destination'],
    },
    {
      id: 'w9-identity-match', label: 'TIN match',
      question: 'Does the page establish an identity-match requirement?',
      query: 'identity_match_required(w9_en)',
      supportedAnswer: 'Yes. Line 1 is required and the TIN must match the name entered there.',
      unsupportedAnswer: 'The reviewed evidence does not establish an identity-match requirement.',
      relatedEvidenceIds: ['w9-en-claim-required-line', 'w9-en-claim-tin-match'],
      expectedStatus: 'answered', expectedSourceRegionIds: ['w9-en-region-line-one', 'w9-en-region-tin'],
      expectedAcceptedClaimIds: ['w9-en-claim-required-line', 'w9-en-claim-tin-match'],
      expectedRuleIds: ['w9-en-rule-identity-match'],
    },
    {
      id: 'w9-deadline', label: 'Filing deadline', question: 'What filing deadline does this page establish?',
      query: 'filing_deadline(w9_en, Date)',
      unsupportedAnswer: 'No accepted evidence on this page establishes a filing deadline.',
      relatedEvidenceIds: ['w9-en-region-purpose', 'w9-en-claim-filing-party'], expectedStatus: 'unsupported',
    },
  ],
};

const IRS_W9_SPANISH: DocumentShowcaseFixture = {
  id: 'irs-w9-spanish', kindLabel: 'Multilingual form', fixtureVersion: 'irs-w9-es-2024-live-v1',
  fileName: 'irs-form-w9-es.pdf', title: 'IRS Form W-9 · Spanish',
  parserMode: 'source_text_reviewed', parserLabel: PARSER_LABEL, adapterLabel: ADAPTER_LABEL,
  source: {
    publisher: 'Internal Revenue Service', url: 'https://www.irs.gov/pub/irs-pdf/fw9sp.pdf',
    retrievedAt: RETRIEVED_AT,
    sha256: 'ee086e2aa83fb662723b5e2641d9487162a7a9fdc153a84df7c88aa5e1ab9d6c',
    pdfPageCount: 6, selectedPageNumbers: [1],
    rightsNote: 'Official Spanish IRS form; retained locally for source-attributed evaluation.',
  },
  defaultQuestionId: 'w9-es-destination',
  pages: [
    {
      id: 'w9-es-page-1', pageNumber: 1, width: 1275, height: 1650,
      label: 'PDF page 1', imageUrl: '/documents/irs-form-w9-es-p1.png',
      imageSha256: '36ce38e97d38a60d4fdae9a19704e6a06b342c25511e365720553c7efe294a41',
      regions: [
        { id: 'w9-es-region-title', pageId: 'w9-es-page-1', pageNumber: 1, order: 1, label: 'Region 1', kind: 'title', text: 'Formulario W-9 — Solicitud y Certificación del Número de Identificación del Contribuyente', bbox: { x: 72, y: 70, width: 902, height: 110 } },
        { id: 'w9-es-region-destination', pageId: 'w9-es-page-1', pageNumber: 1, order: 2, label: 'Region 2', kind: 'status', text: 'Entregue el formulario al solicitante. No lo envíe al IRS.', bbox: { x: 1032, y: 76, width: 174, height: 94 } },
        { id: 'w9-es-region-line-one', pageId: 'w9-es-page-1', pageNumber: 1, order: 3, label: 'Region 3', kind: 'paragraph', text: '1 Nombre de la entidad/del individuo. Se requiere una anotación.', bbox: { x: 120, y: 200, width: 1087, height: 72 } },
        { id: 'w9-es-region-tin', pageId: 'w9-es-page-1', pageNumber: 1, order: 4, label: 'Region 4', kind: 'paragraph', text: 'El TIN tiene que concordar con el nombre provisto en la línea 1 para evitar la retención adicional de impuesto.', bbox: { x: 74, y: 827, width: 790, height: 112 } },
        { id: 'w9-es-region-certification', pageId: 'w9-es-page-1', pageNumber: 1, order: 5, label: 'Region 5', kind: 'paragraph', text: 'Bajo pena de perjurio, yo certifico que el número que aparece en este formulario es mi número de identificación del contribuyente correcto.', bbox: { x: 72, y: 1031, width: 1134, height: 142 } },
      ],
    },
  ],
  claims: [
    { id: 'w9-es-claim-recipient', kind: 'accepted', clause: 'form_recipient(w9_es, solicitante).', summary: 'El destinatario fue aceptado después de revisar el bloque extraído.', pageId: 'w9-es-page-1', pageNumber: 1, regionId: 'w9-es-region-destination', sourceText: 'Entregue el formulario al solicitante. No lo envíe al IRS.' },
    { id: 'w9-es-claim-prohibited-recipient', kind: 'accepted', clause: 'prohibited_recipient(w9_es, irs).', summary: 'La prohibición explícita de enviar el formulario al IRS fue aceptada.', pageId: 'w9-es-page-1', pageNumber: 1, regionId: 'w9-es-region-destination', sourceText: 'Entregue el formulario al solicitante. No lo envíe al IRS.' },
    { id: 'w9-es-claim-required-line', kind: 'accepted', clause: 'required_entry(w9_es, linea_1).', summary: 'La línea 1 requiere una anotación.', pageId: 'w9-es-page-1', pageNumber: 1, regionId: 'w9-es-region-line-one', sourceText: '1 Nombre de la entidad/del individuo. Se requiere una anotación.' },
    { id: 'w9-es-claim-tin-match', kind: 'accepted', clause: 'tin_matches(w9_es, nombre_linea_1).', summary: 'El TIN tiene que concordar con el nombre de la línea 1.', pageId: 'w9-es-page-1', pageNumber: 1, regionId: 'w9-es-region-tin', sourceText: 'El TIN tiene que concordar con el nombre provisto en la línea 1 para evitar la retención adicional de impuesto.' },
    { id: 'w9-es-claim-certification-scope', kind: 'proposed', clause: 'certification_scope(w9_es, taxpayer_identity).', summary: 'The extracted text suggests an identity certification scope, but the broader legal meaning remains proposed.', pageId: 'w9-es-page-1', pageNumber: 1, regionId: 'w9-es-region-certification', sourceText: 'Bajo pena de perjurio, yo certifico que el número que aparece en este formulario es mi número de identificación del contribuyente correcto.' },
  ],
  rules: [
    { id: 'w9-es-rule-destination', clause: 'safe_submission(Form, Recipient) :- form_recipient(Form, Recipient), prohibited_recipient(Form, irs).', summary: 'La ruta aceptada combina el destinatario nombrado con la prohibición de enviar al IRS.', sourceText: 'Regla de enrutamiento revisada.' },
    { id: 'w9-es-rule-identity-match', clause: 'identity_match_required(Form) :- required_entry(Form, linea_1), tin_matches(Form, nombre_linea_1).', summary: 'La anotación obligatoria y la concordancia del TIN establecen el requisito de identidad.', sourceText: 'Regla de cumplimentación revisada.' },
  ],
  questions: [
    { id: 'w9-es-destination', label: '¿Dónde se entrega?', question: '¿A quién se debe entregar el Formulario W-9?', query: 'safe_submission(w9_es, Recipient)', supportedAnswer: 'Se debe entregar al solicitante; no se debe enviar al IRS.', unsupportedAnswer: 'La evidencia revisada no establece un destino seguro.', relatedEvidenceIds: ['w9-es-claim-recipient', 'w9-es-claim-prohibited-recipient'], expectedStatus: 'answered', expectedSourceRegionIds: ['w9-es-region-destination'], expectedAcceptedClaimIds: ['w9-es-claim-recipient', 'w9-es-claim-prohibited-recipient'], expectedRuleIds: ['w9-es-rule-destination'] },
    { id: 'w9-es-identity-match', label: 'Concordancia TIN', question: '¿La página exige que el TIN concuerde con el nombre de la línea 1?', query: 'identity_match_required(w9_es)', supportedAnswer: 'Sí. La línea 1 requiere una anotación y el TIN tiene que concordar con ese nombre.', unsupportedAnswer: 'La evidencia revisada no establece el requisito de concordancia.', relatedEvidenceIds: ['w9-es-claim-required-line', 'w9-es-claim-tin-match'], expectedStatus: 'answered', expectedSourceRegionIds: ['w9-es-region-line-one', 'w9-es-region-tin'], expectedAcceptedClaimIds: ['w9-es-claim-required-line', 'w9-es-claim-tin-match'], expectedRuleIds: ['w9-es-rule-identity-match'] },
    { id: 'w9-es-legal-effect', label: 'Efecto legal', question: '¿Qué efecto legal completo tiene la certificación?', query: 'complete_legal_effect(w9_es, Effect)', unsupportedAnswer: 'El bloque extraído no basta para probar el efecto legal completo de la certificación.', relatedEvidenceIds: ['w9-es-region-certification', 'w9-es-claim-certification-scope'], expectedStatus: 'unsupported' },
  ],
};

const MATHBRIDGE_PAPER: DocumentShowcaseFixture = {
  id: 'mathbridge-paper', kindLabel: 'Research paper', fixtureVersion: 'mathbridge-arxiv-2408-07081-live-v1',
  fileName: 'mathbridge-paper.pdf', title: 'MathBridge · arXiv research paper',
  parserMode: 'source_text_reviewed', parserLabel: PARSER_LABEL, adapterLabel: ADAPTER_LABEL,
  source: {
    publisher: 'arXiv', url: 'https://arxiv.org/pdf/2408.07081', retrievedAt: RETRIEVED_AT,
    sha256: 'f305ae6c54e702c304e3be1ea666c369802c6bf93acd5066327da0d98694cf83',
    pdfPageCount: 9, selectedPageNumbers: [4],
    rightsNote: 'MathBridge is licensed CC BY 4.0; the selected page is redistributed with author, title, source, and license attribution in THIRD_PARTY_NOTICES.md.',
  },
  defaultQuestionId: 'mathbridge-scale',
  pages: [
    {
      id: 'mathbridge-page-4', pageNumber: 4, width: 1275, height: 1650,
      label: 'PDF page 4', imageUrl: '/documents/mathbridge-paper-p4.png',
      imageSha256: '5a0d921fd7ea50f69364d74b7c0089f1b8f6e96d120034aebc8260fb080bf4de',
      regions: [
        { id: 'mathbridge-region-table', pageId: 'mathbridge-page-4', pageNumber: 4, order: 1, label: 'Region 1', kind: 'table_row', text: 'Table 2: Examples of MathBridge dataset. Columns: context_before, LaTeX formula, context_after, spoken_sentence.', bbox: { x: 108, y: 108, width: 1060, height: 330 } },
        { id: 'mathbridge-region-unique-formulas', pageId: 'mathbridge-page-4', pageNumber: 4, order: 2, label: 'Region 2', kind: 'paragraph', text: 'The API generated spoken English text for approximately 13 million unique formulas out of the initial 49 million collected.', bbox: { x: 108, y: 1070, width: 510, height: 258 } },
        { id: 'mathbridge-region-limits', pageId: 'mathbridge-page-4', pageNumber: 4, order: 3, label: 'Region 3', kind: 'paragraph', text: 'MathBridge keeps context before to no more than 200 characters, equation to 80, context after to 250, and spoken English to 120.', bbox: { x: 658, y: 904, width: 520, height: 278 } },
        { id: 'mathbridge-region-retained', pageId: 'mathbridge-page-4', pageNumber: 4, order: 4, label: 'Region 4', kind: 'paragraph', text: 'This process led to the removal of fewer than 500 K data points, maintaining a total of approximately 23 million data points on MathBridge.', bbox: { x: 660, y: 1172, width: 520, height: 184 } },
      ],
    },
  ],
  claims: [
    { id: 'mathbridge-claim-unique', kind: 'accepted', clause: 'unique_formula_count(mathbridge, 13000000).', summary: 'Approximately 13 million unique formulas were processed.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-unique-formulas', sourceText: 'The API generated spoken English text for approximately 13 million unique formulas out of the initial 49 million collected.' },
    { id: 'mathbridge-claim-initial', kind: 'accepted', clause: 'initial_formula_count(mathbridge, 49000000).', summary: 'The initial collection is described as approximately 49 million formulas.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-unique-formulas', sourceText: 'The API generated spoken English text for approximately 13 million unique formulas out of the initial 49 million collected.' },
    { id: 'mathbridge-claim-retained', kind: 'accepted', clause: 'retained_data_points(mathbridge, 23000000).', summary: 'Approximately 23 million data points remained after post-processing.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-retained', sourceText: 'This process led to the removal of fewer than 500 K data points, maintaining a total of approximately 23 million data points on MathBridge.' },
    { id: 'mathbridge-claim-before-limit', kind: 'accepted', clause: 'length_limit(mathbridge, context_before, 200).', summary: 'Reviewed maximum for context_before.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-limits', sourceText: 'MathBridge keeps context before to no more than 200 characters, equation to 80, context after to 250, and spoken English to 120.' },
    { id: 'mathbridge-claim-after-limit', kind: 'accepted', clause: 'length_limit(mathbridge, context_after, 250).', summary: 'Reviewed maximum for context_after.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-limits', sourceText: 'MathBridge keeps context before to no more than 200 characters, equation to 80, context after to 250, and spoken English to 120.' },
    { id: 'mathbridge-claim-table-shape', kind: 'proposed', clause: 'table_schema(mathbridge, four_columns).', summary: 'The extracted table shape is proposed until each cell boundary is reviewed.', pageId: 'mathbridge-page-4', pageNumber: 4, regionId: 'mathbridge-region-table', sourceText: 'Table 2: Examples of MathBridge dataset. Columns: context_before, LaTeX formula, context_after, spoken_sentence.' },
  ],
  rules: [
    { id: 'mathbridge-rule-scale', clause: 'large_dataset(Dataset) :- retained_data_points(Dataset, Count), Count > 20000000.', summary: 'A reviewed threshold classifies a retained dataset above 20 million rows as large.', sourceText: 'Reviewed scale-classification rule for the showcase.' },
    { id: 'mathbridge-rule-context', clause: 'context_before_is_stricter(Dataset) :- length_limit(Dataset, context_before, Before), length_limit(Dataset, context_after, After), Before < After.', summary: 'The context-before limit is stricter when its accepted maximum is lower.', sourceText: 'Reviewed limit-comparison rule.' },
  ],
  questions: [
    { id: 'mathbridge-scale', label: 'Dataset scale', question: 'Does the reviewed page support classifying MathBridge as a large dataset?', query: 'large_dataset(mathbridge)', supportedAnswer: 'Yes. The page reports approximately 23 million retained data points, above the reviewed 20 million threshold.', unsupportedAnswer: 'The reviewed evidence does not establish the dataset scale.', relatedEvidenceIds: ['mathbridge-claim-retained'], expectedStatus: 'answered', expectedSourceRegionIds: ['mathbridge-region-retained'], expectedAcceptedClaimIds: ['mathbridge-claim-retained'], expectedRuleIds: ['mathbridge-rule-scale'] },
    { id: 'mathbridge-context-limit', label: 'Context limits', question: 'Is the context-before length limit stricter than context-after?', query: 'context_before_is_stricter(mathbridge)', supportedAnswer: 'Yes. The accepted limits are 200 characters before and 250 characters after.', unsupportedAnswer: 'The reviewed evidence does not establish the relative limits.', relatedEvidenceIds: ['mathbridge-claim-before-limit', 'mathbridge-claim-after-limit'], expectedStatus: 'answered', expectedSourceRegionIds: ['mathbridge-region-limits'], expectedAcceptedClaimIds: ['mathbridge-claim-before-limit', 'mathbridge-claim-after-limit'], expectedRuleIds: ['mathbridge-rule-context'] },
    { id: 'mathbridge-training-cost', label: 'Training cost', question: 'What did training the dataset cost?', query: 'training_cost(mathbridge, Cost)', unsupportedAnswer: 'No accepted evidence on this page establishes a training cost.', relatedEvidenceIds: ['mathbridge-region-table', 'mathbridge-claim-table-shape'], expectedStatus: 'unsupported' },
  ],
};

const UN_MULTILINGUALISM: DocumentShowcaseFixture = {
  id: 'un-multilingualism', kindLabel: 'Public report', fixtureVersion: 'un-multilingualism-2023-live-v1',
  fileName: 'un-multilingualism.pdf', title: 'UN · Why It Matters: Multilingualism',
  parserMode: 'source_text_reviewed', parserLabel: PARSER_LABEL, adapterLabel: ADAPTER_LABEL,
  source: {
    publisher: 'United Nations Dag Hammarskjöld Library', url: 'https://www.un.org/sites/un2.un.org/files/2227864e.pdf', retrievedAt: RETRIEVED_AT,
    sha256: '3a376c01bb16dc6be32b7465f7a3851746290223073b42193d8f46ea6fc911ed',
    pdfPageCount: 70, selectedPageNumbers: [20], rightsNote: 'United Nations publication; local non-commercial evaluation with source credit.',
  },
  defaultQuestionId: 'un-daily-coverage',
  pages: [
    {
      id: 'un-page-20', pageNumber: 20, width: 1275, height: 1650,
      label: 'PDF page 20 · printed page 15', imageUrl: '/documents/un-multilingualism-p20.png',
      imageSha256: '6a989492d4cef3dbcb868f7e63cea65013eae7e31f4c6f85613108490dae9f04',
      regions: [
        { id: 'un-region-title', pageId: 'un-page-20', pageNumber: 20, order: 1, label: 'Region 1', kind: 'title', text: 'OTHER LANGUAGES AT THE UNITED NATIONS', bbox: { x: 169, y: 126, width: 720, height: 64 } },
        { id: 'un-region-regular-languages', pageId: 'un-page-20', pageNumber: 20, order: 2, label: 'Region 2', kind: 'paragraph', text: 'In addition to the six official languages, other languages, such as Kiswahili and Portuguese, are regularly used by the Department of Global Communications.', bbox: { x: 170, y: 778, width: 990, height: 95 } },
        { id: 'un-region-hindi', pageId: 'un-page-20', pageNumber: 20, order: 3, label: 'Region 3', kind: 'paragraph', text: 'In 2018, Hindi was added to the list of non-official languages receiving daily coverage.', bbox: { x: 170, y: 850, width: 990, height: 72 } },
        { id: 'un-region-resolution', pageId: 'un-page-20', pageNumber: 20, order: 4, label: 'Region 4', kind: 'paragraph', text: 'Resolution 76/268 highlights Portuguese, Hindi, Kiswahili, Persian, Bangla and Urdu in addition to official languages.', bbox: { x: 170, y: 918, width: 990, height: 118 } },
        { id: 'un-region-future', pageId: 'un-page-20', pageNumber: 20, order: 5, label: 'Region 5', kind: 'paragraph', text: 'The Department continues to disseminate information in more non-official languages.', bbox: { x: 170, y: 900, width: 990, height: 62 } },
      ],
    },
  ],
  claims: [
    { id: 'un-claim-official-count', kind: 'accepted', clause: 'official_language_count(un, 6).', summary: 'The page explicitly refers to six official languages.', pageId: 'un-page-20', pageNumber: 20, regionId: 'un-region-regular-languages', sourceText: 'In addition to the six official languages, other languages, such as Kiswahili and Portuguese, are regularly used by the Department of Global Communications.' },
    { id: 'un-claim-kiswahili', kind: 'accepted', clause: 'regular_nonofficial_language(un, kiswahili).', summary: 'Kiswahili is named as a regularly used language beyond the six official languages.', pageId: 'un-page-20', pageNumber: 20, regionId: 'un-region-regular-languages', sourceText: 'In addition to the six official languages, other languages, such as Kiswahili and Portuguese, are regularly used by the Department of Global Communications.' },
    { id: 'un-claim-portuguese', kind: 'accepted', clause: 'regular_nonofficial_language(un, portuguese).', summary: 'Portuguese is named as a regularly used language beyond the six official languages.', pageId: 'un-page-20', pageNumber: 20, regionId: 'un-region-regular-languages', sourceText: 'In addition to the six official languages, other languages, such as Kiswahili and Portuguese, are regularly used by the Department of Global Communications.' },
    { id: 'un-claim-hindi', kind: 'accepted', clause: 'daily_coverage_added(un, hindi, 2018).', summary: 'Hindi was added to daily non-official-language coverage in 2018.', pageId: 'un-page-20', pageNumber: 20, regionId: 'un-region-hindi', sourceText: 'In 2018, Hindi was added to the list of non-official languages receiving daily coverage.' },
    { id: 'un-claim-future-expansion', kind: 'proposed', clause: 'future_coverage_expansion(un, likely).', summary: 'Continued dissemination does not prove a future expansion commitment, so this inference remains proposed.', pageId: 'un-page-20', pageNumber: 20, regionId: 'un-region-future', sourceText: 'The Department continues to disseminate information in more non-official languages.' },
  ],
  rules: [
    { id: 'un-rule-outreach', clause: 'multilingual_outreach(Language) :- regular_nonofficial_language(un, Language).', summary: 'Regularly used non-official languages count as multilingual outreach for this reviewed query.', sourceText: 'Reviewed outreach classification rule.' },
    { id: 'un-rule-coverage', clause: 'documented_daily_coverage(Language, Year) :- daily_coverage_added(un, Language, Year), Year >= 2018.', summary: 'Daily coverage added in or after 2018 is returned as documented coverage.', sourceText: 'Reviewed temporal coverage rule.' },
  ],
  questions: [
    { id: 'un-daily-coverage', label: 'Daily coverage', question: 'Which language does the page say received daily coverage in 2018?', query: 'documented_daily_coverage(Language, Year)', supportedAnswer: 'Hindi was added to daily non-official-language coverage in 2018.', unsupportedAnswer: 'The reviewed page does not establish a daily-coverage addition.', relatedEvidenceIds: ['un-claim-hindi'], expectedStatus: 'answered', expectedSourceRegionIds: ['un-region-hindi'], expectedAcceptedClaimIds: ['un-claim-hindi'], expectedRuleIds: ['un-rule-coverage'] },
    { id: 'un-outreach', label: 'Other languages', question: 'Which regularly used non-official languages are named?', query: 'multilingual_outreach(Language)', supportedAnswer: 'The reviewed page names Kiswahili and Portuguese as regularly used beyond the six official languages.', unsupportedAnswer: 'The reviewed page does not name regularly used non-official languages.', relatedEvidenceIds: ['un-claim-kiswahili', 'un-claim-portuguese'], expectedStatus: 'answered', expectedSourceRegionIds: ['un-region-regular-languages'], expectedAcceptedClaimIds: ['un-claim-kiswahili', 'un-claim-portuguese'], expectedRuleIds: ['un-rule-outreach'] },
    { id: 'un-future-commitment', label: 'Future commitment', question: 'Does this page guarantee expanded language coverage next year?', query: 'guaranteed_future_expansion(un, Year)', unsupportedAnswer: 'No accepted evidence on this page guarantees expanded language coverage next year.', relatedEvidenceIds: ['un-region-future', 'un-claim-future-expansion'], expectedStatus: 'unsupported' },
  ],
};

export const DOCUMENT_SHOWCASE_DEFAULT_FIXTURE_ID = IRS_W9_ENGLISH.id;

export const DOCUMENT_SHOWCASE_FIXTURES: readonly DocumentShowcaseFixture[] = [
  IRS_W9_ENGLISH,
  IRS_W9_SPANISH,
  MATHBRIDGE_PAPER,
  UN_MULTILINGUALISM,
];

export const DOCUMENT_SHOWCASE_FIXTURE = IRS_W9_ENGLISH;
